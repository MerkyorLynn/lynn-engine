# Qwen3.6-35B W4A16 Native MoE P146 V3.2 Resident Probe

**Date:** 2026-05-19  
**Candidate:** `LYNN_NATIVE_ACTIVE_MOE_BACKEND=packed_pretransposed_graphsafe_v32_ordered`  
**Model:** `/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0`

## Verdict

**CLOSED_P37_DRIFT**

The V3.2 ordered scalar path does not pass resident P37 with linear block graph
disabled. It is not eligible for P25 or structured promotion.

## Result

| Probe | Value |
|---|---:|
| graph_on | false |
| P37 exact | 1 / 3 |
| collapse_detected | false |
| baseline TPS range | 26.91-30.18 |
| candidate steady TPS | 33.70-34.01 |

Drifts:

| Prompt | Drift token | Baseline IDs | Candidate IDs |
|---:|---:|---|---|
| 0 | 7 | `[271, 248068, 271, 248069, 271, 24797, 36, 9616]` | `[271, 248068, 271, 248069, 271, 24797, 36, 220]` |
| 1 | 3 | `[198, 727, 51184, 318, 77, 1590, 198, 331]` | `[198, 727, 51184, 1393, 1590, 198, 331, 307]` |

Prompt 2 is exact, but P37 requires 3/3 exact before any escalation.

## Implication

The exact-scalar resident route is not exact enough in real decode despite
looking safer than the fast pretransposed fixture candidates. Native MoE
resident promotion still requires a backend that preserves the Triton numerical
contract at the token-id level before speed gates are meaningful.

## Artifacts

- `benchmarks/p146_resident_moe_backend_p37_probe.py`
- `reports/qwen36_35b/p146_v32_ordered_codex_graphoff_20260519_024406.json`
