# Qwen3.6 Native MoE Output-Owned Validation - 2026-05-18

## Intake Verdict

| Item | Verdict | Why |
|---|---|---|
| KIMI2.6 / `native-moe-output-owned` | ACCEPT as research fixture candidate | R6000 p134 confirms the scheduling win, but it is BF16-dequant only and non-exact. Do not wire into default serving. |
| Claude / `moe-slot-repack-fixture` | PENDING | The branch was not present on `origin` at validation time, so only the written design could be reviewed. Real acceptance requires pushed files plus R6000 p135/p136 reports. |
| Codex graph-safe packed-NVFP4 `_out` ABI | ACCEPT as research artifact | Caller-owned scratch removes the worst CUDA graph capture failure mode and preserves the speed signal, but P37 still drifts. |

## KIMI2.6 Output-Owned BF16

Source branch: `origin/claude/native-moe-output-owned-20260518`.

Main only cherry-picked the minimal candidate files from the stale branch:

- `benchmarks/candidates/native_output_owned_bf16.py`
- `csrc/lynn_native/moe_output_owned_bf16.cu`
- `reports/qwen36_35b/native_output_owned_bf16_report.md`

R6000 p134 routed-only result:

| Metric | Value |
|---|---:|
| Decision | `FAST_CANDIDATE` |
| Passed | 18/18 relaxed |
| Exact | 0/18 |
| max_abs_max | 3.90625e-3 |
| rel_l2_max | 6.82867e-3 |
| cosine_min | 0.999980211 |
| candidate_ms_mean | 0.0504708 ms |
| speedup vs p134 fixture reference | 18.36x |

Interpretation: the output-owned down-stage scheduling idea is real. It should influence the packed NVFP4 path, but this BF16 implementation is not production-serving material because it consumes dequantized BF16 weights and is non-exact.

## Packed NVFP4 Graph-Safe `_out` ABI

This validation adds a caller-owned scratch variant:

- `active_moe_grouped_per16_nonatomic_out_reference(...)`
- runtime backend `LYNN_NATIVE_ACTIVE_MOE_BACKEND=grouped_per16_nonatomic_out`

R6000 p134 routed-only relaxed:

| Metric | Value |
|---|---:|
| Decision | `FAST_CANDIDATE` |
| Passed | 18/18 relaxed |
| Exact | 0/18 |
| max_abs_max | 3.90625e-3 |
| rel_l2_max | 7.19833e-3 |
| cosine_min | 0.999975026 |
| candidate_ms_mean | 0.0531687 ms |
| speedup vs p134 fixture reference | 16.89x |

R6000 P37 graph-on:

| Metric | Baseline | Candidate |
|---|---:|---:|
| median decode TPS | 104.43 | 129.74 |
| mean decode TPS | 104.38 | 117.46 |
| median speedup | - | 1.242x |
| exact-greedy | - | RED |

The previous native-owned scratch path could collapse into token-id 0 / `!` repetition under graph capture. The caller-owned `_out` ABI no longer shows that graph-capture failure signature, but greedy output still drifts into repeated `think` / prompt text. This narrows the problem from "graph unsafe" to "packed native numeric drift crosses generation margins."

## P33 Layer Triage

Full native candidate:

- first top-1 divergence: step 2, Triton top1 `1393`, candidate top1 `318`
- first layer below threshold: step 0, layer 5, cosine `0.999997139`, max_abs `4.8828125e-4`

Single-layer and family probes did not hit top-1 divergence in 4 steps:

| Layer spec | P33 top1 pass | First layer drift |
|---|---:|---|
| `full_attention` | true | step 0, layer 5 |
| `linear_attention` | true | step 0, layer 15 |
| `0` | true | step 2, layer 2 |
| `4` | true | step 1, layer 7 |
| `5` | true | step 1, layer 7 |
| `8` | true | none |
| `16` | true | none |
| `20` | true | step 3, layer 22 |
| `28` | true | step 3, layer 30 |
| `32` | true | step 3, layer 34 |
| `36` | true | none |
| `39` | true | none |

The generation failure appears cumulative, not caused by one isolated toxic layer. That makes a "native only selected layers" promotion unlikely to be enough; the next useful work is lowering packed NVFP4 drift itself.

## Claude Slot-Repack Acceptance Gate

`claude/moe-slot-repack-fixture-20260518` was not found on `origin` during validation:

```text
fatal: couldn't find remote ref claude/moe-slot-repack-fixture-20260518
```

Required before intake:

1. Push `claude/moe-slot-repack-fixture-20260518` to GitHub.
2. Run `bash scripts/r6000_qwen36_moe_slot_repack.sh` on R6000.
3. Provide p135/p136 JSON outputs showing 18/18 GREEN.
4. If p136 is GREEN, continue with `native_slot_output_owned_bf16` against the p136 fixture format.

## Next Work

1. Keep the safe default unchanged at W4A16 NVFP4 Triton active MoE.
2. Treat packed native `_out` as a speed-proven but quality-blocked branch.
3. Push the next kernel work toward output-owned packed NVFP4 with lower drift, not more service-loop toggles.
4. Use slot-repack fixtures as the next isolated target once Claude's branch is pushed and validated on R6000.
