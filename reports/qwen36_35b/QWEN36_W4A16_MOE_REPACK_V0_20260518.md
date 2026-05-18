# Qwen3.6 35B W4A16 MoE Repack V0

Date: 2026-05-18

## Result

| Item | Value |
|---|---:|
| Sidecar layers | 40 |
| Sidecar size | 18.8624 GiB |
| Build elapsed | 25.593 s |
| P127 contract | GREEN |
| Contract layers | 40/40 |

The sidecar is a MoE-only serving layout. Each layer file co-locates router, active gate/up, active down, shared expert, and shared gate tensors. Active expert tensors are stored as expert-major 3D tensors:

- `active_gate_up.packed`: `[256, 1024, 1024]`
- `active_down.packed`: `[256, 2048, 256]`
- `router.weight`: `[256, 2048]`

## Paths

- R6000 sidecar: `/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-moe-repack-v0`
- Manifest: `reports/qwen36_35b/qwen36_w4a16_moe_repack_manifest_20260518.json`
- P127 all-layer contract: `reports/qwen36_35b/p127_moe_repack_sidecar_contract_all40_20260518.json`

## Next

The next kernel-boundary task should consume this sidecar directly instead of chasing generic manifest keys. The first ABI target is a strict active-MoE boundary that preserves the current math order: router -> top-k -> gate/up -> BF16 inter -> down -> weighted sum -> shared add.
