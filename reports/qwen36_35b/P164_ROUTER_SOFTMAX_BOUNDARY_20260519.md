# P164 Router Softmax Boundary Probe

Date: 2026-05-19

## Candidate

`LYNN_ROUTER_SOFTMAX_OUT_BUFFER=1`

This follows P163's promoted `torch.topk(..., out=...)` path and tests whether
`torch.softmax(..., dtype=float32, out=scratch)` helps the decode router
boundary.

## R6000 Fixture Result

Artifact: `reports/qwen36_35b/p164_qwen36_router_softmax_boundary_probe_20260519.json`

| Metric | Value |
|---|---:|
| Fixtures | 18 |
| Exact ids/routes/logits | 18 / 18 |
| topk-out router mean | 0.039006 ms |
| topk+softmax-out router mean | 0.039062 ms |
| router delta | +0.000056 ms |
| softmax-only alloc mean | 0.012348 ms |
| softmax-only out mean | 0.010427 ms |
| softmax-only delta | -0.001920 ms |
| Decision | `ROUTER_SOFTMAX_OUT_BUFFER_CLOSED_OR_FLAT` |

## Interpretation

The softmax output buffer is numerically exact and the isolated softmax-only
microbench is faster.  However, the full router boundary is flat on the
18-fixture mean because the saved allocation is below measurement noise once
`F.linear` and `torch.topk(..., out=...)` are included.

Do not promote this knob by itself.  It may be bundled into a later prepared
router wrapper if that wrapper has a measurable P25 gain.
