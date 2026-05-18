# P166 Shared Torch In-Place Rerun

Date: 2026-05-19

## Candidate

```text
LYNN_SHARED_EXPERT_GATE_BACKEND=torch_inplace
LYNN_MOE_ADD_SHARED_INPLACE=1
```

This keeps the shared expert math in Torch but mutates the shared output and
MoE output buffers in place.

## Promotion Gate

| Gate | Result |
|---|---:|
| P37 exact | true |
| P37 median speedup | 1.0013x |
| P25 512 decode TPS | 107.28 |
| Hard structured | 40/40 |
| Hard structured mean decode TPS | 107.53 |
| Decision | closed, below safe default |

## Decision

Do not promote.  The candidate is numerically safe on P37 and structured
format, but the end-to-end serving path is flat to slightly slower than the
current safe default.  The shared finalize tail is not the next useful speed
lever without a larger fused boundary.

## Artifact

- `reports/qwen36_35b/r6000_qwen36_w4a16_shared_torch_inplace_20260519_0710_shared_torch_inplace_promotion_summary.json`
