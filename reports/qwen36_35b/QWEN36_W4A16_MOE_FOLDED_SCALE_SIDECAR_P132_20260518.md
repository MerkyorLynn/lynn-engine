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

## Decision

Keep the folded-scale sidecar path as a native-kernel input format.  It is a
small but clean repack improvement: active scale/global division is removed
offline while preserving the current W4A16 math contract.  The sidecar loader
now exposes folded active scales as `_gate_up_effective_scale` and
`_down_effective_scale`; when those aliases are present, the resident runner can
skip runner-time scale replacement.  Full 40-layer folded sidecar generation
should wait until a native grouped kernel consumes this format; the one-layer
probe is enough to validate the contract.

## Artifacts

- `scripts/qwen36_w4a16_moe_repack_sidecar.py --fold-active-global-scale`
- `benchmarks/p127_moe_repack_sidecar_contract.py`
- `benchmarks/p128_moe_repack_triton_boundary_probe.py`
- `reports/qwen36_35b/p132_moe_folded_scale_sidecar_contract_layer0_20260518.json`
- `reports/qwen36_35b/p132_moe_folded_scale_sidecar_triton_boundary_layer0_20260518.json`
