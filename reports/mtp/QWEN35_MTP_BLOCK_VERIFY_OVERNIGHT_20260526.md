# Qwen35 MTP Block Verify Overnight Status

Date: 2026-05-26 00:45 Asia/Shanghai

Morning update: 2026-05-26 09:35 Asia/Shanghai

Post-reboot smoke update: 2026-05-26 10:15 Asia/Shanghai

Active repo status update: 2026-05-27 Asia/Shanghai

This line is active, not abandoned. The 5/20 product pivot still leaves Lynn
engine as an R&D path, and this branch is now the concrete R&D path: port the
portable Nemotron-style runtime skeleton into Qwen35/Qwen9B APEX-MTP. A concise
public-facing status note is tracked at:

```text
reports/mtp/LYNN_ENGINE_ACTIVE_RESEARCH_STATUS_20260527.md
```

## Target

Make the Nemotron-style verify/accept/crop runtime useful for Lynn's own Qwen
line, with the same implementation surface available to:

- Qwen3.6-35B-A3B with the existing official MTP sidecar.
- Qwen3.5-9B once a compatible draft head or trained sidecar exists.

The key split is deliberate: runtime ABI first, draft-head quality second. If
K>1 accept quality is the blocker, it becomes a training task, not a runtime
rewrite.

Recommended product decision after the parallel Nemotron research branch:

- Pick **A+B**.
- A: keep production work on Lynn K=N + APEX-MTP expansion.
- B: run a cheap Qwen3.5 4B LoRA PoC later to measure the diffusion/self-spec
  accept ceiling before paying for Qwen3.5 9B continued pretraining.
- Park Qwen3.6-35B-A3B diffusion training for now; MoE plus bidirectional
  diffusion is too much uncertainty for the first training dollar.

## Local Branch

Branch:

```text
codex/qwen35-mtp-block-verify
```

Changed files:

```text
engine/full_forward.py
engine/incremental_decode.py
engine/mtp_serving.py
engine/mtp_sidecar.py
engine/resident_runner.py
scripts/a100_pack_lynn_native_nvfp4.py
scripts/spark_mtp_speculative_smoke.py
scripts/spark_wait_qwen35_mtp_smoke.sh
reports/mtp/QWEN35_MTP_BLOCK_VERIFY_OVERNIGHT_20260526.md
```

Local syntax gate:

```text
python3 -m compileall -q engine/full_forward.py engine/incremental_decode.py engine/mtp_serving.py engine/mtp_sidecar.py engine/resident_runner.py scripts/a100_pack_lynn_native_nvfp4.py scripts/spark_mtp_speculative_smoke.py
```

Status: passed.

## Runtime Work Completed

Implemented a shared K=N speculative verification skeleton:

- `decode_full_attn_block(...)`: prefix-causal block verifier for full-attention layers.
- `decode_linear_attn_block(...)`: sequential Gated Delta-Net rollout for block verification.
- `_decode_layer_block(...)`: layer-level K=N decode path, delegating K=1/K=2 to existing hot paths.
- `decode_block_to_logits_and_hidden(...)`: runs `[pending, draft_1, ..., draft_k]` and returns all-position logits.
- `speculative_step_kn_batched(...)`: verify/accept/replay rollback path for K>1.
- `LYNN_MTP_SPECULATIVE_K`: opt-in env knob; default behavior remains K=1.
- Smoke report fields for `draft_tokens_proposed`, `accepted_draft_tokens`, and `draft_accept_rate`.
- Conservative K>N state repair: even full-block accepts restore the pre-block
  recurrent/conv snapshot and replay the committed prefix through canonical T=1
  decode before continuing. This protects token exactness while the K>N verifier
  is still experimental.

The K>1 draft source is intentionally marked experimental: the current official
sidecar is one-token MTP. Recursive chaining lets the runtime be tested, but it
does not prove the head was trained for multi-token self-drafting.

## Spark Artifact Recovery

Spark no longer had the historical Lynn-native W4A16 Qwen35 artifact:

```text
/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000
```

Rebuilt it from the existing BF16 Qwen35 checkpoint using the Lynn-native packer.
The original packer OOM-killed Spark because it quantized whole tensors at once.
Patched `scripts/a100_pack_lynn_native_nvfp4.py` with row-chunk quantization.

New artifact:

```text
/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526
```

Pack result:

```json
{
  "output_shards": 7,
  "quantized_count": 553,
  "kept_count": 492,
  "elapsed_seconds": 400.4462103843689
}
```

Validation:

```text
size: 23G
layout: lynn_native_per16_variable
backend: lynn_native_per16
manifest: lynn-variable-nvfp4-pack-v1
```

This restores the W4A16 input needed for Lynn MTP experiments.

## Spark Smoke State

Attempted a minimal real-model smoke:

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e LYNN_MOE_IMPL=packed_nvfp4 \
  -e LYNN_MOE_FAST_FIXED=1 \
  -e LYNN_NATIVE_ACTIVE_MOE_BACKEND=triton \
  -e LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode \
  -e LYNN_NATIVE_DOWN_BACKEND=triton \
  -e LYNN_ROUTER_TOPK_SORTED=0 \
  -e LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare \
  -e LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1 \
  -e LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1 \
  -e LYNN_LINEAR_STATE_UPDATE=inplace \
  -e LYNN_LINEAR_BLOCK_GRAPH=0 \
  -e LYNN_LINEAR_BLOCK_GRAPH_REUSE=0 \
  -e LYNN_LINEAR_BLOCK_GRAPH_PREWARM=0 \
  -e LYNN_PACKED_DECODE=0 \
  -e LYNN_PACKED_SHARED_EXPERT=0 \
  -e LYNN_NATIVE_FP4_LM_HEAD=1 \
  -e LYNN_QK_NORM_ROPE_BACKEND=triton_pair \
  -e LYNN_RMSNORM_GATED_BACKEND=triton \
  -e LYNN_FULL_ATTN_ROPE_CACHE=1 \
  -v /home/merkyor/lynn-engine:/workspace \
  -v /home/merkyor/models:/models \
  -w /workspace \
  lynn-eval-base:cu13 \
  python3 scripts/spark_mtp_speculative_smoke.py \
    --model /models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526 \
    --sidecar /models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors \
    --out reports/mtp/qwen35_mtp_k4_smoke_20260526_0018.json \
    --max-new 8 \
    --spec-k-list 4
```

Observed:

- The runner loaded via Lynn variable NVFP4 slow-dequant path first.
- This is the existing Lynn resident design: BF16 resident weights plus packed
  aliases for decode kernels.
- Current Spark GPU was already occupied by Nemotron SGLang (~52GB), a
  llama-server (~30GB), and ASR/TTS services. The smoke entered unhealthy memory
  pressure and SSH banner exchange started timing out.

Conclusion: do not treat this failed smoke as an algorithm result. It is a
resource contention/run-path result.

Morning connectivity diagnosis:

- `dgx` / `dgx-spark` via Tencent frp `127.0.0.1:2224`: connection refused.
- `dgx-via-ssh` via Tencent autossh `127.0.0.1:2222`: connection refused.
- Tencent currently listens only on N5 reverse port `127.0.0.1:2223`.
- `dgx-via-n5` reaches TCP port 22 on `192.168.100.26`, but SSH banner times
  out from N5. This points to Spark sshd/host load, not a local code issue.
- Last local sync to Spark happened before the conservative K>N replay repair
  and Config-D-ish smoke env refinements, so those final fixes still need rsync
  once SSH recovers.

Post-reboot result:

- Spark SSH recovered after host reboot, and the latest engine/scripts were
  synced to `/home/merkyor/lynn-engine`.
- Qwen35 W4A16 + official MTP sidecar smoke ran successfully on Spark.
- APEX llama.cpp service was temporarily stopped during the 35B Python runner
  smoke to avoid swap pressure, then restarted after the run.

Primary artifact:

```text
reports/mtp/qwen35_mtp_kn_strict_20260526_100603.json
```

Strict verifier smoke means full-attention K=2/K=block uses the canonical T=1
primitive internally. This is the correctness-first default until the true
batched full-attention verifier is bit-stable.

Summary:

| Config | Exact vs baseline | TPS | Accept | Draft accept |
|---|---:|---:|---:|---:|
| baseline | 100% | 33.65 | n/a | n/a |
| shadow | 100% | 34.31 | n/a | n/a |
| spec_k1 | 100% | 29.58 | 74.04% | 74.04% |
| spec_k1_batched_strict | 100% | 29.25 | 77.16% | 77.16% |
| spec_k2_batched_strict | 100% | 16.29 | 2.38% | 63.34% |
| spec_k4_batched_strict | 100% | 13.72 | 0.00% | 45.14% |

Control artifact without strict full-attention fallback:

```text
reports/mtp/qwen35_mtp_kn_smoke_retry_20260526_095832.json
```

That run reproduced the runtime gap:

- `spec_k1` sequential stayed exact.
- `spec_k1_batched`, `spec_k2_batched`, and `spec_k4_batched` failed token
  exactness when the fast batched full-attention verifier was enabled.
- Therefore the current blocker is batched full-attention verifier numerical
  parity, not MTP draft quality, KV crop, or reject replay.

Code default was changed accordingly:

- `LYNN_FULL_ATTN_K2_BACKEND` now defaults to `t1_loop`.
- `LYNN_FULL_ATTN_BLOCK_BACKEND` now defaults to `t1_loop`.
- Fast paths remain available only by explicit env opt-in:
  `LYNN_FULL_ATTN_K2_BACKEND=k2` and `LYNN_FULL_ATTN_BLOCK_BACKEND=block`.

## Late-Night K2 Fast-Path Diagnosis

Follow-up sweep after the strict smoke isolated the fast K2 drift more tightly.
The full-attention K2 fast path first diverges at layer 3 even at zero advance:

| Artifact | Extra env | Safe combos | Finding |
|---|---|---|---|
| `qwen35_m18_layer_sweep_20260526_223235.json` | default `triton_pair` QK/RoPE | `t1_full_attn_only`, `t1_both` | Fast `decode_full_attn_k2` drifts at L3 full-attention. |
| `qwen35_m18_layer_sweep_qk_torch_20260526_223856.json` | `LYNN_QK_NORM_ROPE_BACKEND=torch` | `t1_full_attn_only`, `t1_both` | Drift remains, so it is not just Triton-pair vs torch QK/RoPE. |
| `qwen35_m18_layer_sweep_rowwise_t1_20260526_224505.json` | `LYNN_FULL_ATTN_K2_PROBE=rowwise_t1` | `t1_full_attn_only`, `t1_both` | Row-wise attention/gate/o-proj alone reduces one position but does not restore exactness. |
| `qwen35_m18_layer_sweep_rowwise_qkv_20260526_225400.json` | `LYNN_FULL_ATTN_K2_PROBE=rowwise_qkv` | `t1_full_attn_only`, `t1_both` | Row-wise QKV/RoPE alone also does not restore exactness. |
| `qwen35_m18_layer_sweep_rowwise_qkv_t1_20260526_230101.json` | `LYNN_FULL_ATTN_K2_PROBE=rowwise_qkv_rowwise_t1` | all four combos | Exact parity returns: logits and recurrent state diffs are zero in the M18 probe. |

Conclusion: the K2 verifier mismatch is a composition effect across batched
QKV/RoPE plus batched attention/gate/o-proj, not KV crop, reject replay,
training, or linear-attention state handling. A correctness bridge now exists:

```text
LYNN_FULL_ATTN_K2_BACKEND=rowwise_bridge
```

This mode keeps the K2 verifier shell but computes full-attention QKV/RoPE and
attention/o-proj row-wise inside `decode_full_attn_k2`. It is a diagnostic
parity bridge, not a throughput solution.

End-to-end bridge smoke:

```text
reports/mtp/qwen35_mtp_safe_k2_bridge_20260526_230739.json
```

| Config | Exact vs baseline | TPS | Accept | Draft accept |
|---|---:|---:|---:|---:|
| baseline | 100% | 34.29 | n/a | n/a |
| spec_k1 | 100% | 29.20 | 74.04% | 74.04% |
| spec_k1_batched | 100% | 28.92 | 77.16% | 77.16% |
| spec_k2_batched + rowwise bridge | 100% | 16.27 | 2.38% | 63.34% |

The bridge confirms correctness but does not improve speed over the strict
T=1 fallback. The next performance task is therefore a real batched full-attn
K2 kernel/implementation that matches T=1 numerics, not more row-wise fallback.

Additional SDPA isolation on 2026-05-27 tested `manual_gqa` for both T=1 and
K2 full-attention decode:

```text
reports/mtp/qwen35_m18_layer_sweep_manual_gqa_20260527_000731.json
reports/mtp/qwen35_mtp_k2_manual_gqa_fast_20260527_001547.json
```

Layer sweep result:

| Combo | Result |
|---|---|
| `k2_both` + `manual_gqa` | drift, first bad L5 `linear_attention` |
| `t1_full_attn_only` + `manual_gqa` | exact |
| `t1_linear_attn_only` + `manual_gqa` | drift, first bad L5 `linear_attention` |
| `t1_both` + `manual_gqa` | exact |

End-to-end smoke result with `LYNN_FULL_ATTN_DECODE_BACKEND=manual_gqa`,
`LYNN_FULL_ATTN_K2_BACKEND=k2`, full-accept fast commit, and prefix repair:

| Config | Exact vs baseline | TPS | Accept | Draft accept |
|---|---:|---:|---:|---:|
| baseline | 100% | 33.66 | n/a | n/a |
| spec_k1 | 100% | 29.39 | 77.16% | 77.16% |
| spec_k1_batched | 66.67% | 31.54 | 80.05% | 80.05% |
| spec_k2_batched | 83.33% | 25.90 | 62.40% | 69.30% |

Conclusion: replacing SDPA with the existing Python `manual_gqa` path is not
enough. It improves some accept/TPS numbers but breaks token-exactness in the
batched verifier. The remaining K2 parity work must separate batched
projection/RoPE, attention, gating, and `o_proj` numerics rather than treating
SDPA as the only source of drift.

Two component probes were added to split the exact row-wise bridge:

```text
LYNN_FULL_ATTN_K2_PROBE=rowwise_qkv_rowwise_attn_batched_o
LYNN_FULL_ATTN_K2_PROBE=rowwise_qkv_batched_attn_rowwise_o
```

`scripts/spark_mtp_m18_k2_layer_sweep_probe.py` now also supports
`--probe-mode-list`, so future component sweeps can reuse one model load instead
of relaunching the 35B container for each probe.

Artifacts:

```text
reports/mtp/qwen35_m18_layer_sweep_rowwise_qkv_rowwise_attn_batched_o_20260527_002523.json
reports/mtp/qwen35_m18_layer_sweep_rowwise_qkv_batched_attn_rowwise_o_20260527_003147.json
```

Component result:

| Probe | Meaning | Result |
|---|---|---|
| `rowwise_qkv_rowwise_attn_batched_o` | QKV/RoPE + attention row-wise; gate/`o_proj` batched | drift at L5 |
| `rowwise_qkv_batched_attn_rowwise_o` | QKV/RoPE + gate/`o_proj` row-wise; attention batched | drift at L5 |
| `rowwise_qkv_rowwise_t1` | QKV/RoPE + attention + gate/`o_proj` all row-wise | exact |

So the fast K2 problem has at least two independent numerical contributors:
batched attention and batched output projection/gating. A production fast K2
path must make both components T=1-equivalent, or choose a controlled
approximate mode as a deliberate product/model change.

Follow-up split of gate vs `o_proj`:

```text
reports/mtp/qwen35_m18_layer_sweep_gate_o_split_20260527_004529.json
```

| Probe | Meaning | Result |
|---|---|---|
| `rowwise_qkv_rowwise_attn_batched_gate_rowwise_o` | only gate is batched; `o_proj` row-wise | exact |
| `rowwise_qkv_rowwise_attn_rowwise_gate_batched_o` | gate row-wise; only `o_proj` batched | drift at L5 |

This isolates the output side further: batched sigmoid/gating is safe, while
the `[2, hidden] @ o_proj.weight` batched matmul is not T=1-equivalent enough
for deterministic MTP parity. The remaining independent blocker is batched
attention itself. The next fast-K2 kernel work should therefore target:

1. A dual-row attention kernel that preserves T=1-equivalent accumulation.
2. A dual-row `o_proj` kernel/dispatch that shares weight loads but accumulates
   each row exactly like the T=1 path.

The safe component split is now named:

```text
LYNN_FULL_ATTN_K2_BACKEND=rowwise_gate_bridge
```

It maps to QKV/RoPE row-wise, attention row-wise, gate batched, and `o_proj`
row-wise. This is expected to be only marginally faster than `rowwise_bridge`,
but it captures the strongest proven exact bridge: the gate multiply can be
batched; `o_proj` and attention cannot yet.

A tiny no-model reproduction now exists:

```text
scripts/spark_k2_linear_row_parity_probe.py
reports/mtp/qwen35_k2_linear_row_parity_warm_20260527_005716.json
scripts/spark_k2_attention_row_parity_probe.py
reports/mtp/qwen35_k2_attention_row_parity_20260527_005902.json
```

On Spark BF16, random `[1, 2, 4096] @ [4096, 4096].T` differed from two
separate `[1, 1, 4096]` calls in **32/32** seeds. Warm timings were roughly
0.08 ms batched vs 0.27 ms row-wise, which explains the temptation to batch
`o_proj`; the deterministic verifier cannot use that shortcut until a
T=1-equivalent dual-row kernel exists.

The attention fixture compares K2 prefix-causal batched SDPA against two
row-wise SDPA calls at `[Hq=32, Hkv=4, N=2048, D=128]`. It differed in
**16/16** seeds. On this shape, batched SDPA with `attn_mask + enable_gqa`
was also slower than row-wise SDPA (about 1.08 ms vs 0.08 ms), so the current
PyTorch batched attention route is neither exact nor a useful speed path.

One low-level `o_proj` direction was prototyped:

```text
triton_kernels/rowwise_linear.py
scripts/spark_k2_rowwise_linear_kernel_probe.py
reports/mtp/qwen35_k2_rowwise_linear_kernel_20260527_020517.json
```

The prototype uses independent per-token accumulators inside one Triton launch.
On Spark BF16 `[2, 4096] @ [4096, 4096].T`, K2 single-launch output matched two
T1 kernel launches in **8/8** seeds, with mean timings around 0.18 ms vs
0.75 ms. It does not match PyTorch T1 bit-for-bit, so it is not a drop-in for
the current production baseline. But it proves the useful contract: if both
baseline T1 and K2 verifier route `o_proj` through the same rowwise-linear
kernel, the output-projection half of K2 can be exact while saving launches.

The rowwise-linear kernel is now wired behind an opt-in full-attention `o_proj`
backend:

```text
LYNN_FULL_ATTN_O_PROJ_BACKEND=rowwise_triton
```

Spark sanity confirmed the helper returns bit-equal output for one K2 call vs
two T1 calls (`max_abs=0`). This is still experimental: it changes the
accumulation contract from PyTorch/cuBLAS, so it must be enabled for both the
baseline T1 path and K2 verifier when measuring token exactness.

The matching dual-row prefix-attention direction now has a lightweight kernel
probe too:

```text
triton_kernels/rowwise_attention.py
scripts/spark_k2_rowwise_attention_kernel_probe.py
reports/mtp/qwen35_k2_rowwise_attention_kernel_20260527_023349.json
reports/mtp/qwen35_k2_rowwise_attention_kernel_stride_20260527_023632.json
```

On Spark BF16 `[Hq=32, Hkv=4, N=2048, D=128]`, the K2 single-launch kernel
matched two T1 kernel launches in **8/8** seeds with `max_abs=0`. Warmed timing
was about **0.077 ms** for K2 vs **0.123 ms** for two T1 calls after removing
unnecessary K/V contiguous copies from the wrapper. This is the
first positive signal that the attention half can be made T=1-equivalent
without paying the full row-wise launch cost. It is still a micro-kernel result;
the next gate is wiring it into the real `decode_full_attn`/`decode_full_attn_k2`
path and running a Qwen35 maintenance smoke.

The kernel has now been wired behind opt-in experimental knobs:

```text
LYNN_FULL_ATTN_ATTENTION_BACKEND=rowwise_triton
LYNN_FULL_ATTN_K2_BACKEND=rowwise_kernel_bridge
```

Real Qwen35 maintenance smoke:

```text
reports/mtp/qwen35_mtp_k2_rowwise_attention_kernel_warm_20260527_024429.json
```

Run knobs:

```text
LYNN_FULL_ATTN_ATTENTION_BACKEND=rowwise_triton
LYNN_FULL_ATTN_O_PROJ_BACKEND=rowwise_triton
LYNN_FULL_ATTN_K2_BACKEND=rowwise_kernel_bridge
LYNN_MTP_KN_FULL_ACCEPT_FAST_COMMIT=1
LYNN_MTP_KN_PREFIX_BLOCK_REPAIR=1
WARMUP_RUNS=2
```

Result:

| Config | Exact vs baseline | TPS | Accept | Draft accept |
|---|---:|---:|---:|---:|
| baseline | 100% | 27.82 | n/a | n/a |
| spec_k1 | 100% | 22.41 | 68.33% | 68.33% |
| spec_k1_batched | 100% | 10.87 | 68.33% | 68.33% |
| spec_k2_batched | 100% | 22.99 | 61.11% | 61.11% |

Interpretation: the rowwise attention kernel improves the previous exact K2
bridge from 21.28 TPS to 22.99 TPS (**+8.1%**) while preserving 6/6 token
exactness, but it still lands at **0.83x** of the warmed T1 baseline under the
same experimental backend. This proves the extracted verify/accept/crop flow
and the K2 attention kernel are directionally useful for APEX-MTP, but it is
not a production speedup yet. Remaining cost is now likely split across
row-wise QKV/RoPE projection, rowwise `o_proj`, MTP sidecar overhead, and the
Python runner/service loop.

Follow-up dynamic-N kernel update:

```text
reports/mtp/qwen35_k2_rowwise_attention_kernel_dynamicn_20260527_025345.json
reports/mtp/qwen35_mtp_k2_rowwise_attention_dynamicn_warm_20260527_025502.json
```

The first kernel used `N` as a Triton constexpr, which can compile separate
variants for different sequence lengths and pollute prompt-level baseline
timings. The current kernel passes `N` at runtime and loops with `while n0 < N`.
Micro speed is slower (**0.115 ms** K2 vs **0.162 ms** for two T1 calls at
`N=2048`), but different prompt lengths no longer create the same misleading
baseline/JIT artifact.

Current-code real smoke:

| Config | Exact vs baseline | TPS | Accept | Draft accept |
|---|---:|---:|---:|---:|
| baseline | 100% | 37.03 | n/a | n/a |
| spec_k1 | 100% | 25.74 | 68.33% | 68.33% |
| spec_k1_batched | 100% | 20.16 | 68.33% | 68.33% |
| spec_k2_batched | 100% | 23.00 | 61.11% | 61.11% |

This is the cleanest current answer to "does the extracted algorithm improve
APEX-MTP?": correctness and accept/reject/crop semantics are proven on Qwen35,
and the K2 attention micro-kernel is directionally useful, but the complete
Python runner K2 path is still **0.62x** of the warmed baseline. The next
optimization target is not the control flow anymore; it is verifier cost:
row-wise QKV/RoPE, rowwise `o_proj`, MTP sidecar launch/forward, and Python
loop overhead.

## ROI Profiling Pass

Instead of guessing the next kernel, an opt-in profiler was added:

```text
LYNN_MTP_PROFILE=1
LYNN_MTP_PROFILE_SYNC=1
```

It resets per `runner.generate()` and writes an `mtp_profile` section into the
smoke JSON. The smoke wrapper now also supports a small `PROMPTS_JSON` file for
low-risk profile runs.

Profile artifact:

```text
reports/mtp/qwen35_mtp_k2_profile_20260527_093355.json
reports/mtp/qwen35_mtp_profile_prompts.json
```

Two-prompt profile result:

| Config | Exact vs baseline | TPS | Accept | Draft accept |
|---|---:|---:|---:|---:|
| baseline | 100% | 34.63 | n/a | n/a |
| spec_k1 | 100% | 23.88 | 75.00% | 75.00% |
| spec_k1_batched | 100% | 17.86 | 75.00% | 75.00% |
| spec_k2_batched | 100% | 23.79 | 66.67% | 66.67% |

The profiler changes the ROI target:

| Section | Signal |
|---|---|
| `spec_k2_batched` / `block_verify.layers_total` | About 70.8 ms/event. This is a 3-token block (`pending + 2 drafts`) and still mostly falls back to T1 verifier work. |
| `spec_k1_batched` / `k1_batched.k2_forward` | About 101.6 ms/event in the profile run, including one compile outlier. |
| `k2_layer.moe` | Large steady cost. K2 currently calls the packed NVFP4 decode MoE once per token because the packed path is T=1-only. |
| `full_attn.attention.rowwise_triton` | Small after the new kernel; attention is no longer the main local bottleneck. |
| `kn.draft_chain` | About 5.3 ms/event for two chained drafts; not the first optimization target. |

Highest-ROI shortcut test:

```text
LYNN_MTP_K2_MOE_MODE=batched_optimized
reports/mtp/qwen35_mtp_k2_moe_batched_optimized_20260527_094126.json
```

Result: **do not promote.**

| Config | Exact vs baseline | TPS | Notes |
|---|---:|---:|---|
| spec_k1_batched + BF16 optimized batched MoE | 0% | 12.30 | Exactness failed and speed got worse. |
| spec_k2_batched | 100% | 24.50 | Unchanged generic block path, not proof that batched MoE is safe. |

Conclusion: BF16 optimized batched MoE is not a safe shortcut. If Lynn Python
runner is to beat its baseline, the next real engine task is a **packed NVFP4
T=2 MoE verifier kernel** that matches the T1 packed accumulation contract.
Otherwise the higher-ROI production move is to port the already-correct
verify/accept/crop flow into the llama.cpp/APEX service loop and avoid Python
dispatch entirely.

Operational note: the production APEX service is configured with
`Restart=always`, so a plain `systemctl stop lynn-apex-mtp-llamacpp.service`
will be undone during long 35B Python smoke loads. Any future maintenance smoke
must use a temporary runtime mask/stop wrapper and always unmask/start on exit.
The live APEX-MTP fallback was healthy after this diagnosis; one observed
production request reported 512-token eval at about 66.7 tok/s with draft
acceptance around 0.469.

Maintenance-window smoke with the new rowwise `o_proj` contract:

```text
reports/mtp/qwen35_mtp_k2_rowwise_gate_real2_20260527_021806.json
```

Run knobs:

```text
LYNN_FULL_ATTN_O_PROJ_BACKEND=rowwise_triton
LYNN_FULL_ATTN_K2_BACKEND=rowwise_gate_bridge
LYNN_MTP_KN_FULL_ACCEPT_FAST_COMMIT=1
LYNN_MTP_KN_PREFIX_BLOCK_REPAIR=1
```

Result:

| Config | Exact vs baseline | TPS | Accept | Draft accept |
|---|---:|---:|---:|---:|
| baseline | 100% | 31.74 | n/a | n/a |
| spec_k1 | 100% | 24.24 | 66.67% | 66.67% |
| spec_k1_batched | 100% | 22.99 | 66.67% | 66.67% |
| spec_k2_batched | 100% | 21.28 | 53.33% | 53.70% |

Interpretation for "does the extracted algorithm improve APEX MTP?":

- Correctness: **yes**. The verify/accept/crop/full-accept/prefix-repair
  control flow is token-exact on the real Qwen35 W4A16 + official MTP sidecar.
- Throughput today: **no**. Even with rowwise `o_proj`, K2 is only 0.67x of
  the Python baseline on this short smoke. The high-accept prompt reaches
  34.30 tok/s, still below its 37.18 tok/s baseline row.
- Production relevance: the active llama.cpp APEX-MTP fallback is already much
  faster than this Python runner (observed about 66.7 tok/s). The extracted
  algorithm is therefore not a drop-in production speedup yet; it needs
  T=1-equivalent dual-row attention and `o_proj` kernels, or integration into
  the production llama.cpp APEX path.

The non-strict fast-K2 control does not justify accepting approximate drift:

```text
reports/mtp/qwen35_mtp_kn_smoke_retry_20260526_095832.json
```

| Config | Exact vs baseline | TPS | Notes |
|---|---:|---:|---|
| spec_k1_batched fast-K2 | 33.33% | 29.47 | Similar speed to strict, but token exactness breaks. |
| spec_k2_batched fast-K2 | 83.33% | 16.59 | Only +1.8% over strict K2, still far below baseline. |
| spec_k4_batched fast block | 50.00% | 14.27 | Approximate and still slow. |

So the shortcut "ship approximate fast-K2 as a slightly different model" is not
attractive: it loses deterministic parity without buying meaningful TPS.

## K>N Full-Accept Fast Commit

The first real K>N speed recovery is to avoid replay when the verifier accepts
the entire proposed block. This is gated by:

```text
LYNN_MTP_KN_FULL_ACCEPT_FAST_COMMIT=1
```

Behavior:

- If `accepted_count == draft_count`, keep the verifier's final state and use
  the last verifier row's hidden/logits for `next_base_hidden` and
  `next_pending_id`.
- If the block is partially accepted or rejected, keep the existing conservative
  restore-and-replay path.

Spark smoke:

```text
reports/mtp/qwen35_mtp_kn_full_accept_fast_20260526_233834.json
```

| Config | Exact vs baseline | TPS | Accept | Draft accept |
|---|---:|---:|---:|---:|
| baseline | 100% | 33.42 | n/a | n/a |
| spec_k1 | 100% | 29.29 | 74.04% | 74.04% |
| spec_k1_batched | 100% | 28.99 | 77.16% | 77.16% |
| spec_k2_batched + full-accept fast commit | 100% | 24.36 | 56.57% | 63.80% |

Compared with the strict K2 replay smoke, K2 effective TPS improves from
16.29 to 24.36 tok/s (**+49.5%**) while preserving 6/6 token exactness.
The mean is still below baseline, but high-accept prompts already cross it:
prompt 003 reaches 35.42 tok/s. This confirms the runtime now has the right
shape: better draft training directly converts into throughput instead of
being swallowed by replay.

K4 was tested with the same optimization:

```text
reports/mtp/qwen35_mtp_k4_full_accept_fast_20260526_234710.json
```

Result: **do not promote K4 direct commit yet.**

| Config | Exact vs baseline | TPS | Accept | Draft accept |
|---|---:|---:|---:|---:|
| spec_k4_batched + full-accept fast commit | 83.33% | 15.51 | 19.12% | 44.38% |

One prompt diverged, and mean TPS remains far below baseline. The runtime now
defaults the fast-commit cap to K=2 via:

```text
LYNN_MTP_KN_FULL_ACCEPT_FAST_COMMIT_MAX_K=2
```

K>2 needs separate block-state parity work before direct state commit is safe.

## K2 Partial-Accept Prefix Repair

The next replay reduction is partial-accept repair for K=2. When exactly one
draft token is accepted, the committed prefix is `[pending, accepted_draft]`.
The new opt-in path restores the pre-block recurrent/conv snapshot, then
repairs that committed prefix with a smaller exact block verifier instead of
falling back to canonical token-by-token T=1 replay.

Knobs:

```text
LYNN_MTP_KN_PREFIX_BLOCK_REPAIR=1
LYNN_MTP_KN_PREFIX_BLOCK_REPAIR_MAX_LEN=2
```

Spark smoke:

```text
reports/mtp/qwen35_mtp_k2_prefix_repair_20260526_235627.json
```

| Config | Exact vs baseline | TPS | Accept | Draft accept |
|---|---:|---:|---:|---:|
| baseline | 100% | 33.43 | n/a | n/a |
| spec_k1 | 100% | 29.24 | 74.04% | 74.04% |
| spec_k1_batched | 100% | 28.94 | 77.16% | 77.16% |
| spec_k2_batched + full-accept fast commit + prefix repair | 100% | 24.79 | 58.16% | 65.10% |

This keeps the K2 gate at 6/6 exact and nudges K2 effective TPS from 24.36 to
24.79 tok/s over the full-accept-only smoke. The improvement is small, so keep
this path opt-in for now. The more important result is that K2 direct-state
commit and K2 prefix-state repair both preserve token parity when capped to
two-token blocks.

## Next Smoke Command

Run once Spark is free enough, ideally after Nemotron SGLang exits:

```bash
docker run --rm --gpus all --ipc=host \
  -v /home/merkyor/lynn-engine:/workspace \
  -v /home/merkyor/models:/models \
  -w /workspace \
  lynn-eval-base:cu13 \
  python3 scripts/spark_mtp_speculative_smoke.py \
    --model /models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526 \
    --sidecar /models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors \
    --out reports/mtp/qwen35_mtp_k4_smoke_$(date +%Y%m%d_%H%M%S).json \
    --max-new 32 \
    --spec-k-list 2,4
```

Or use the non-destructive waiter, which does not stop any service:

```bash
MODEL=/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526 \
SIDECAR=/home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors \
MAX_USED_MIB_BEFORE_RUN=60000 \
MAX_NEW=32 \
SPEC_K_LIST=2,4 \
scripts/spark_wait_qwen35_mtp_smoke.sh
```

Success criteria:

- Baseline runs without resource pressure.
- `spec_k1` and `spec_k1_batched` remain token-exact vs baseline.
- `spec_k2_batched`/`spec_k4_batched` either pass token-exact or fail with a
  specific traceback/prefix boundary.
- If token-exact passes but `draft_accept_rate` falls off with K, the next step
  is draft-head training, not verifier rewrite.

## Training Path If Needed

If the official one-token sidecar does not chain cleanly to K=4/K=8, train the
draft side rather than forcing runtime tricks:

1. Start with Qwen35-A3B K=2/K=4 MTP continuation training.
2. Freeze the base model initially; train an offset-k draft adapter/head against
   teacher-forced base continuations.
3. Use heldout prompt continuations to measure accept rate, not just CE loss.
4. Promote only if `draft_accept_rate` plus block-verify cost beats graph
   baseline TPS.
5. Mirror the same ABI for Qwen3.5-9B with a dense-model sidecar.

Hardware estimate:

- Dual A100 is enough for adapter/head training and accept-rate experiments.
- H100 shortens iteration time and makes longer K/longer context batches easier.
- Full joint AR-diffusion continued pretrain is a separate, much larger project;
  start with targeted MTP/block-draft training first.
