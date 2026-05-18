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

Next useful work is to keep the output-owned/non-atomic scheduling shape, then reduce drift by aligning the Triton contract more closely:

1. preserve the existing Triton two-stage contract and only replace the down output-owned reduction;
2. compare candidate against stored `routed_output` and a Triton-intermediate dump to isolate whether drift enters at gate/up or down;
3. only after the isolated stage is strict or bounded, wire it into P37/P25.
