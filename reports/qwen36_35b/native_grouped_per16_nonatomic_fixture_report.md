# Native Grouped-Per16 Non-Atomic Fixture Report

Date: 2026-05-18
Machine: RTX PRO 6000 Blackwell
Scope: p134 routed-only fixture gate for the packed-NVFP4 native-owned scratch path.

## Verdict

`native_grouped_per16_nonatomic` is a fast research candidate, not a strict default candidate.

It proves the current packed-NVFP4 one-boundary admission path can run in the same latency class as the BF16 output-owned prototype, but it is not exact against the p134 routed reference.

## Results

| Gate | Threshold | Result | candidate_ms_mean | max_abs_max | rel_l2_max | cosine_min |
|---|---|---:|---:|---:|---:|---:|
| strict | max_abs <= 0, cosine >= 0.999999 | CLOSED_NUMERIC, 0/18 | 0.051269 ms | 3.90625e-3 | 7.1983e-3 | 0.999975026 |
| relaxed | max_abs <= 0.02, cosine >= 0.99997, allow nonexact | FAST_CANDIDATE, 18/18 | 0.052450 ms | 3.90625e-3 | 7.1983e-3 | 0.999975026 |

Artifacts:

- `reports/qwen36_35b/p134_routed_only_native_grouped_per16_nonatomic_20260518_grouped_per16_nonatomic_strict_v2.json`
- `reports/qwen36_35b/p134_routed_only_native_grouped_per16_nonatomic_20260518_grouped_per16_nonatomic_strict_v2.summary.json`
- `reports/qwen36_35b/p134_routed_only_native_grouped_per16_nonatomic_20260518_grouped_per16_nonatomic_relaxed.json`
- `reports/qwen36_35b/p134_routed_only_native_grouped_per16_nonatomic_20260518_grouped_per16_nonatomic_relaxed.summary.json`

## Interpretation

This narrows the problem: packed NVFP4 native-owned scratch is not slow at fixture scale; it is blocked by numerical contract drift. The drift shape closely matches the BF16 output-owned candidate, which points toward accumulation order / BF16 intermediate rounding rather than an obvious route or weight-layout bug.

P135 stage isolation sharpened this further:

| Stage | max_abs_max | rel_l2_max | Mean Triton ms | Mean native ms |
|---|---:|---:|---:|---:|
| gate/up native vs Triton intermediate | 3.0518e-5 | 5.8762e-6 | 0.032850 | 0.030037 |
| down native vs Triton on Triton intermediate | 1.5259e-5 | 8.2954e-5 | 0.028433 | 0.020822 |
| native full vs stored fixture reference | 3.90625e-3 | 7.1983e-3 | n/a | n/a |

That means the native pieces are close to Triton at the isolated active-MoE stage; the larger fixture drift is mostly the existing Triton-vs-PyTorch fixture-reference difference.

However, a real P37 generate gate still closes this backend:

| Gate | Result |
|---|---|
| P37 graph-on exact | RED |
| graph-on candidate decode TPS mean | 134.41 |
| graph-on median speedup | 1.249x |
| graph-on failure shape | first token ok, then token id 0 / `!` repetition |
| P33 graph-off top1 follow | PASS for 3 steps |
| P37 graph-off exact | RED |
| graph-off candidate decode TPS mean | 29.89 |
| graph-off median speedup | 0.306x |

So this is the first clear 130+ TPS speed shape, but it is not promotable until the runtime contract is fixed.

Artifacts:

- `reports/qwen36_35b/p135_moe_native_stage_drift_20260518.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_grouped_per16_nonatomic_stagechecked_20260518_stagechecked_p37_p37.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_grouped_per16_nonatomic_stagechecked_20260518_stagechecked_p37_promotion_summary.json`
- `reports/qwen36_35b/p33_grouped_per16_nonatomic_first_divergence_20260518.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_grouped_per16_nonatomic_graphoff_20260518_grouped_nonatomic_graphoff_p37_p37.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_grouped_per16_nonatomic_graphoff_20260518_grouped_nonatomic_graphoff_p37_promotion_summary.json`

Next useful work is to keep the output-owned/non-atomic scheduling shape, then fix the runtime contract:

1. make the native MoE boundary CUDA-graph-capture safe, or exclude it from linear-block graph capture while keeping the rest of the graph reusable;
2. preserve the existing Triton two-stage contract and first replace only the down output-owned reduction;
3. only after P37 exact is fixed, wire the native path into P25/structured again.
