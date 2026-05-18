# Qwen3.6 35B W4A16 MoE Folded-Scale Sidecar Probe

Date: 2026-05-18

## Purpose

P132 moves the P130 effective-scale idea into the offline MoE sidecar format.
When `--fold-active-global-scale` is passed to
`scripts/qwen36_w4a16_moe_repack_sidecar.py`, active expert scales are stored as
`scale / global_scale`, and the active global-scale tensors are stored as one.

This is not a serving promotion candidate by itself.  It gives the next native
grouped MoE kernel a direct effective-scale input contract without requiring
runner-time scale replacement.

## Layer-0 R6000 Probe

| Check | Result |
|---|---:|
| folded sidecar build | GREEN |
| sidecar size, layer 0 | 578.9 MiB |
| P127 contract | GREEN |
| P128 Triton boundary | GREEN |
| P128 max abs / mean abs | 0.0 / 0.0 |
| folded effective aliases | gate/up true, down true |
| manifest active-MoE timing | 0.08716 ms |
| folded sidecar active-MoE timing | 0.08210 ms |
| manifest / sidecar ratio | 1.062x |

## All-40 R6000 Probe

The full folded-scale sidecar has now been built and validated for every
language layer.

| Check | Result |
|---|---:|
| folded sidecar path | `/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-moe-repack-folded-scale-v0` |
| layers | 40 |
| sidecar size | 23 GiB on disk |
| files | 40 layer files + manifest |
| P127 contract | GREEN, 40/40 |
| P128 Triton boundary | GREEN, 40/40 |
| P128 max abs / mean abs | 0.0 / 0.0 |
| manifest active-MoE mean | 0.08317 ms/layer |
| folded sidecar active-MoE mean | 0.08244 ms/layer |
| manifest / sidecar mean ratio | 1.009x |

## Decision

Keep the folded-scale sidecar path as a native-kernel input format.  It is a
small but clean repack improvement: active scale/global division is removed
offline while preserving the current W4A16 math contract.  The sidecar loader
now exposes folded active scales as `_gate_up_effective_scale` and
`_down_effective_scale`; when those aliases are present, the resident runner can
skip runner-time scale replacement.  The full all-layer sidecar is now a
validated ABI for Stream A native grouped-MoE work.  It is not expected to move
TPS by itself; its value is removing scale/global handling and manifest-layout
variance before replacing the inner active-MoE math.

## Artifacts

- `scripts/qwen36_w4a16_moe_repack_sidecar.py --fold-active-global-scale`
- `benchmarks/p127_moe_repack_sidecar_contract.py`
- `benchmarks/p128_moe_repack_triton_boundary_probe.py`
- `reports/qwen36_35b/p132_moe_folded_scale_sidecar_contract_layer0_20260518.json`
- `reports/qwen36_35b/p132_moe_folded_scale_sidecar_triton_boundary_layer0_20260518.json`
- `reports/qwen36_35b/p132_moe_folded_scale_sidecar_contract_all40_20260518.json`
- `reports/qwen36_35b/p132_moe_folded_scale_sidecar_triton_boundary_all40_20260518.json`
- `scripts/qwen36_candidate_env_moe_folded_sidecar.env`
