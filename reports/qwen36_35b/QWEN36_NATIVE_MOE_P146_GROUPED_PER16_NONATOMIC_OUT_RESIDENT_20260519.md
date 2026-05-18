# Qwen3.6-35B W4A16 Native MoE P146 Nonatomic-Out Resident Probe

**Date:** 2026-05-19  
**Candidate:** `LYNN_NATIVE_ACTIVE_MOE_BACKEND=grouped_per16_nonatomic_out`  
**Extra env:** `LYNN_MOE_ACTIVE_SCRATCH=1`  
**Model:** `/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0`

## Verdict

**CLOSED_P37_DRIFT**

The caller-owned scratch / nonatomic-out resident path does not pass P37 with
linear block graph disabled. It is not eligible for P25 or structured promotion.

## Result

| Probe | Value |
|---|---:|
| graph_on | false |
| P37 exact | 2 / 3 |
| collapse_detected | false |
| baseline TPS range | 25.85-29.25 |
| candidate TPS range | 10.41-32.65 |

Prompt-level result:

| Prompt | Exact | Drift token | Baseline IDs | Candidate IDs |
|---:|---|---:|---|---|
| 0 | true | - | `[271, 248068, 271, 248069, 271, 24797, 36, 9616]` | `[271, 248068, 271, 248069, 271, 24797, 36, 9616]` |
| 1 | true | - | `[198, 727, 51184, 318, 77, 1590, 198, 331]` | `[198, 727, 51184, 318, 77, 1590, 198, 331]` |
| 2 | false | 2 | `[271, 248068, 271, 248069, 271, 37586, 1679, 9616]` | `[271, 248068, 198, 8160, 579, 264, 7047, 1817]` |

## Implication

The graph-safe scratch ABI removes the token-0 collapse risk seen in earlier
graph-captured native paths, but the resident generation still drifts at token
2 on one P37 prompt. This keeps `grouped_per16_nonatomic_out` in the research
bucket. The next native MoE step should either exactly preserve the Triton
active path's accumulation/rounding contract or move up a larger boundary while
keeping the Triton numerical contract intact.

## Artifacts

- `benchmarks/p146_resident_moe_backend_p37_probe.py`
- `reports/qwen36_35b/p146_grouped_per16_nonatomic_out_codex_graphoff_20260519_024646.json`
