# P130 Effective-Scale Rerun After Router Top-K Promotion

Date: 2026-05-19

## Purpose

Re-test the MoE effective-scale path after the safe default moved to
`LYNN_ROUTER_TOPK_OUT_BUFFER=1`.

Candidate env:

```text
LYNN_MOE_REPACK_SIDECAR_DIR=/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-moe-repack-v0
LYNN_MOE_EFFECTIVE_SCALE=1
```

## Local Active-MoE Probe

The local active-MoE boundary remains strictly equivalent and faster.

| Metric | Result |
|---|---:|
| Exact | 9/9 |
| Max abs | 0.0 |
| Max rel L2 | 0.0 |
| Min cosine | 0.99999994 |
| Mean reference active | 0.05530 ms |
| Mean effective active | 0.05085 ms |
| Active speedup | 1.0876x |

## Promotion Gate

| Gate | Result |
|---|---:|
| P37 exact | true |
| P37 median speedup | 1.0094x |
| P25 512 decode TPS | 105.94 |
| Hard structured | 40/40 |
| Hard structured mean decode TPS | 108.10 |
| Decision | closed, below safe default |

## Decision

Keep effective-scale as an exact local building block, not a default serving
profile.  It lowers the active-MoE micro floor, but full serving with the
sidecar path does not beat the current safe default.

## Artifacts

- `reports/qwen36_35b/p130_moe_effective_scale_probe_20260519_0655.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_moe_effective_scale_rerun_20260519_0658_moe_eff_routertopk_promotion_summary.json`
