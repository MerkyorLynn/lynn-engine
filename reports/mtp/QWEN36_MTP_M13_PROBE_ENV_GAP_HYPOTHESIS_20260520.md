# Qwen3.6-35B MTP M13 Followup — Probe-vs-Smoke Env Gap

**Date:** 2026-05-20
**Context:** [M13 result](QWEN36_MTP_M13_FULLATTN_T1LOOP_RESULT_20260520.md) shows accept-rate parity (75.17% ≈ 75.13%) after `LYNN_FULL_ATTN_K2_BACKEND=t1_loop`, but batched still 2/6 exact / 65 mean-prefix. Codex's note: "one additional verifier mismatch remains."

## Finding: the strict diff probe is not measuring the production env

The strict probe ([commit `26f36d5`](https://github.com/MerkyorLynn/lynn-engine/commit/26f36d5)) reports `first_bad_layer: null` and final-logit `max_abs=0.0`. **But its [`BASE_ENV`](../../scripts/spark_mtp_k2_vs_t1_diff_probe.py) is missing four production flags that the M13 smoke does set**:

| flag | probe `BASE_ENV` | smoke `BASE_ENV` |
|---|---|---|
| `LYNN_PACKED_DECODE` | unset | `1` |
| `LYNN_PACKED_SHARED_EXPERT` | unset | `1` |
| `LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4` | unset | `1` |
| `LYNN_FULL_ATTN_QKV_FUSED` | unset | `1` |

Without `LYNN_PACKED_DECODE=1`, `_decode_weight` ([engine/incremental_decode.py:77-91](../../engine/incremental_decode.py)) returns the BF16 weight under `w[key]` instead of the packed alias at `w[key + ".packed"]`. Both T=1 `decode_full_attn` and the K=2 t1-loop fallback then run `F.linear` on BF16 weights — identical kernel, identical math, hence `max_abs=0.0`. The probe is currently certifying BF16-fallback equivalence, not production-fast-path equivalence.

## Per-prompt batched detail (M13)

| # | Prompt (head) | Prefix | Exact | Accept |
|---|---|---:|---:|---:|
| 0 | "Explain the difference between Q4_K_M and NVFP4 …" | 39 | ✗ | 61.3% |
| 1 | "用一句话解释 speculative decoding 的核心思想。" | 36 | ✗ | 45.7% |
| 2 | "Write a Python function … Fibonacci…" | 128 | ✓ | 89.7% |
| 3 | "If a train travels 60 mph for 2.5 hours…" | 77 | ✗ | 91.0% |
| 4 | "请输出一个 JSON: {\"city\": …}" | 27 | ✓ | 86.7% |
| 5 | "Summarize the role of the MoE router…" | 83 | ✗ | 76.7% |

Prompt #3 has 91.0% accept but 77/100 prefix — argmax-level mismatch hits a single position then everything downstream diverges. Prompts that pass exact (#2, #4) either terminate early (#4: 27 tokens) or have very deterministic continuations (#2: Fibonacci).

## Recommended next probe iteration

Re-run the K2-vs-T1 strict diff probe with the four missing flags exported:

```
LYNN_PACKED_DECODE=1 \
LYNN_PACKED_SHARED_EXPERT=1 \
LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1 \
LYNN_FULL_ATTN_QKV_FUSED=1 \
LYNN_FULL_ATTN_K2_BACKEND=t1_loop \
python3 -u scripts/spark_mtp_k2_vs_t1_diff_probe.py …
```

If `first_bad_layer` surfaces under this env, the layer type + position points to which production fast path diverges between the T=1 sequential chain and the K=2 t1_loop sub-call chain. Most likely candidates:

1. **`quantize_fp4_m1_native` per-row branch** (M11, [engine/resident_runner.py:910-928](../../engine/resident_runner.py)): `for row in h2d: quantize_fp4_m1_native(row.contiguous())` iterates a 2D tensor and yields **1D** rows of shape `[K]`, not `[1, K]`. If scale derivation in `quantize_fp4_m1_native` differs by an ULP for 1D vs 2D input, K=2 verify logits drift slightly each round.
2. **`decode_linear_attn` `LYNN_PACKED_DECODE_LINEAR_ATTN` path**: in `_in_proj_qkv_z_b_a.weight` vs four separate `_linear` calls, the production env routes one direction; the fused fast kernel may pick a different autotune config when called from `decode_linear_attn_k2` (which slices an already-allocated `[:, t:t+1, :].contiguous()` tensor) than when called from sequential decode (which receives the runner's own `[1, 1, D]` buffer).
3. **`LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1`** state thread vs assign: the K=2 path inside `decode_linear_attn_k2` threads `recurrent_state, conv_state` between two T=1 calls as local variables, while the sequential T=1 path goes through `state.update_linear_attn_state(...)` → `state.recurrent_state[layer_idx] = new` between calls. Result-value-wise identical, but if any downstream layer reads `state.recurrent_state[layer_idx]` mid-block in the K=2 path, identity-vs-value semantics diverge.

## Why this matters

Strict-diff round-1 cosine 1.0 plus 2/6 exact end-to-end means the divergence is either (a) **not detected by a single-prompt single-round probe** because the env is wrong, or (b) **only emerges after multiple speculative rounds** (cumulative argmax tie-break under epsilon drift). Either way, the existing probe verdict alone shouldn't gate the next round of fixes; running the probe under the same env that the smoke uses is the cheap test that distinguishes (a) from (b).

If the production-env probe still reports `first_bad_layer: null` then the divergence is genuinely multi-round and the next fix targets state-snapshot semantics, argmax tie-break determinism, or the per-row `quantize_fp4_m1_native` call shape. If it reports a `first_bad_layer`, the layer type names the next fix target directly.

## Cross-reference

- [memory `project_lynn_engine_t1_only_kernel_contract_20260519`](../../../../memory/project_lynn_engine_t1_only_kernel_contract_20260519.md) — the unifying "T=1-only kernel" contract that frames every fix in this chain (M9 / M11 / M12 / M13).
- [M13 result](QWEN36_MTP_M13_FULLATTN_T1LOOP_RESULT_20260520.md) — canonical metrics.
- [M12 lm_head opt-in result](QWEN36_MTP_M12_LMHEAD_OPTIN_RESULT_20260520.md)
- [K2 strict diff result](QWEN36_MTP_K2_T1FULL_STRICT_DIFF_RESULT_20260520.md)
