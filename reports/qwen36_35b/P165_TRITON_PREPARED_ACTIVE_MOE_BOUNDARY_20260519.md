# P165 Qwen3.6-35B Triton-Prepared Active MoE Boundary

Date: 2026-05-19

## Purpose

P159 showed the active-MoE inter/out scratch path was exact but mostly flat,
while the hot boundary still carried redundant Python-side prepared-tensor
checks and wrapper conversions. P165 adds an opt-in prepared Triton boundary
that reuses the exact same gate/up and down Triton kernels while requiring the
resident path to pass already-contiguous prepared tensors and caller-owned
scratch:

```text
LYNN_MOE_ACTIVE_SCRATCH=1
LYNN_MOE_TRITON_PREPARED=1
LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode
LYNN_NATIVE_DOWN_BACKEND=triton
```

This is an exactness-first boundary probe, not a default promotion.

## Gate Result

| Gate | Result |
|---|---:|
| P37 exact | true |
| P37 median speedup | 1.0068x |
| P25 512 decode TPS | 106.64 |
| Hard structured | 40/40 |
| Hard structured mean decode TPS | 107.41 |
| Decision | CLOSED |

## Readout

The prepared boundary is numerically safe: P37 matched all three greedy
generations exactly. It does not improve the service line enough to promote.
P25 512 falls below the current safe default threshold, and the 40-request
structured gate is correct but not faster.

Keep the code behind `LYNN_MOE_TRITON_PREPARED=1` as a reusable exact boundary
artifact. Do not enable it by default. The useful next step is not another
wrapper tweak; it is a larger exact MoE/GDN boundary that removes real kernel
launches or changes the offline layout.

## Artifacts

- `scripts/qwen36_candidate_env_triton_prepared.env`
- `reports/qwen36_35b/r6000_qwen36_w4a16_triton_prepared_20260519_1158_triton_prepared_p37.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_triton_prepared_20260519_1158_triton_prepared_p25.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_triton_prepared_20260519_1158_triton_prepared_hard_structured.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_triton_prepared_20260519_1158_triton_prepared_promotion_summary.json`
