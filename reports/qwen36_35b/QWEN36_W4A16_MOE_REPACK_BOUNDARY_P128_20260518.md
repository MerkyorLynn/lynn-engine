# Qwen3.6 35B W4A16 MoE Repack Boundary Probe

Date: 2026-05-18

P128 feeds the current Triton active-MoE boundary directly from the MoE repack sidecar and compares it with the old manifest-loaded tensors.

| Metric | Value |
|---|---:|
| Layers | 40/40 |
| Result | GREEN |
| Max abs diff | 0.0 |
| Sidecar active-MoE mean | 0.082569 ms |
| Manifest active-MoE mean | 0.083620 ms |

This confirms the repack v0 sidecar is a valid kernel-input ABI. It does not yet fuse gate/up and down into a new native kernel; it removes layout uncertainty before that work starts.

Artifacts:

- `benchmarks/p128_moe_repack_triton_boundary_probe.py`
- `reports/qwen36_35b/p128_moe_repack_triton_boundary_all40_20260518.json`
