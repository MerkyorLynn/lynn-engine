# Qwen3.6-35B W4A16 Native MoE P152 Native Packed vs P147

**Date:** 2026-05-19  
**Candidate:** `moe_slot_packed_nvfp4_probe`  
**Fixture source:** `/root/autodl-tmp/reports/qwen36_35b/p138_packed_slot_fixtures_kimi_20260518`

## Verdict

P152 is **closed for resident promotion** but useful for implementation
direction.

It exports native packed-MoE candidate outputs and compares them against the new
P147 Triton-stage contract. The candidate is close and fast, but not exact:

| Metric | Value |
|---|---:|
| P147 exact fixtures | 12/18 |
| Max output abs error | 1.2207e-4 |
| Candidate latency mean | 0.09036 ms |
| P147 Triton gate/up mean | 0.07541 ms |
| P147 Triton down mean | 0.02915 ms |
| P147 Triton stage total | 0.10456 ms |

The native candidate is slightly faster than the two-stage Triton fixture budget,
but it fails the exact contract. Since resident P37 has already shown that
sub-millithreshold MoE errors can roll into greedy token drift, this remains a
research candidate only.

## Non-Exact Fixtures

| Layer | Prompt | max_abs | cosine |
|---:|---:|---:|---:|
| 4 | 0 | 9.31e-10 | 0.99999994 |
| 20 | 0 | 7.63e-6 | 1.00000000 |
| 28 | 0 | 1.53e-5 | 1.00000000 |
| 39 | 0 | 1.91e-6 | 1.00000000 |
| 8 | 1 | 1.22e-4 | 0.99999994 |
| 36 | 1 | 1.19e-7 | 1.00000000 |

## Readout

The useful finding is that the current native packed path is no longer a
grossly wrong math path. It is close to the Triton-stage contract and already
has a fixture-stage speed edge. The next kernel work should focus on eliminating
the tiny reduction/rounding differences against P147, not on another broad
backend sweep.

## Next Step

Add a stage-diff probe for the six non-exact fixtures:

1. compare gate/up intermediate against P147 `triton_inter`;
2. compare down output with native inter versus Triton inter;
3. isolate whether the remaining error comes from FP4 decode order, SiLU
   approximation, down reduction order, or final BF16 store.

Only a P147 exact candidate should be escalated to P146/P37 again.

## Artifacts

- `benchmarks/p152_native_packed_moe_stage_outputs.py`
- `scripts/r6000_qwen36_moe_p152_native_to_p147.sh`
- `reports/qwen36_35b/p152_native_packed_outputs_20260519_0432.json`
- `reports/qwen36_35b/p152_native_packed_vs_p147_20260519_0432.json`
