# Qwen3.6-35B P159 Triton Active Boundary Probe

**Date:** 2026-05-19  
**Model:** official Qwen3.6-35B-A3B Lynn-native W4A16 NVFP4  
**Probe:** `benchmarks/p159_qwen36_triton_active_boundary_probe.py`

## Result

P159 compares the exact P157-style Triton active-MoE stage against the same
Triton kernels with caller-owned `inter` and `out` scratch tensors.

Artifact:
`reports/qwen36_35b/p159_qwen36_triton_active_boundary_probe_20260519_0505.json`

| Metric | Value |
|---|---:|
| Fixtures | 18 |
| Scratch inter exact | 18 / 18 |
| Scratch out exact | 18 / 18 |
| Allocating active mean | 0.05267 ms |
| Caller-owned scratch mean | 0.05251 ms |
| Delta | -0.00016 ms |

## Interpretation

Caller-owned scratch is numerically safe and useful as graph/fusion plumbing,
but it is not a standalone TPS lever. The effect is below noise and confirms
the earlier P144 lesson: scratch ownership alone does not move the serving
profile.

Next 35B work should prioritize exact shared-finalize and router/top-k boundary
coarsening rather than more active-stage allocation probes.
