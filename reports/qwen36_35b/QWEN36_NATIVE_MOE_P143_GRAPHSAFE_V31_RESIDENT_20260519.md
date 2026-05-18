# Qwen3.6-35B W4A16 Native MoE P143 Resident Probe

**Date:** 2026-05-19  
**Candidate:** `LYNN_NATIVE_ACTIVE_MOE_BACKEND=packed_pretransposed_graphsafe_v31`  
**Model:** `/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0`

## Verdict

**CLOSED_P37_DRIFT**

The V3.1 graph-safe pretransposed MoE path does not pass resident P37 even with
linear block graph disabled. It is not eligible for P25 or structured promotion.

## Result

| Probe | Value |
|---|---:|
| graph_on | false |
| P37 exact | 2 / 3 |
| collapse_detected | false |
| baseline TPS range | 26.41-29.55 |
| candidate steady TPS | 11.68-11.69 |

Prompt 1 drifts at token 3:

| Field | Token IDs |
|---|---|
| baseline | `[198, 727, 51184, 318, 77, 1590, 198, 331]` |
| candidate | `[198, 727, 51184, 1393, 1590, 198, 331, 307]` |

The first candidate prompt reports only `0.15` TPS because the path lazily
dequantizes/gathers/pretransposes selected expert weights on the first live
decode call. Subsequent prompts are still only about `11.7` TPS, far below the
safe default line.

## Implication

This closes the current cuBLAS/pretransposed graph-safe V3.1 resident route as a
promotion candidate. The next useful Native MoE work should avoid dynamic
per-token dequant/gather in the resident path and must first prove strict P37
exactness before any P25 or structured gates.

## Artifacts

- `reports/qwen36_35b/p143_graphsafe_v31_codex_graphoff_20260519_023838.json`
