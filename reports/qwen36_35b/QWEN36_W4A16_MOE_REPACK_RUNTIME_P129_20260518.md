# Qwen3.6 35B W4A16 MoE Repack Runtime Gate - 2026-05-18

## Result

The MoE sidecar is now wired into the resident runtime behind an explicit
opt-in env:

```bash
LYNN_MOE_REPACK_SIDECAR_DIR=/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-moe-repack-v0
```

R6000 verified that this path attaches all 40 active-MoE layers from the
sidecar and preserves generation parity. It is a valid kernel-input ABI, but
not a speed breakthrough by itself.

| Candidate | P37 exact | P25 512 decode TPS | Hard structured | Runtime attach | Decision |
|---|---:|---:|---:|---:|---|
| `moe_repack_sidecar` | true | 107.39 | 40/40, mean 107.74 TPS | 40 layers | research-only |
| `moe_repack_scratch` | true | 107.08 | 40/40, mean 107.39 TPS | 40 layers + 40 scratch | research-only |

## Interpretation

The sidecar and scratch boundary are strict:

- `P37` exact-greedy parity passes.
- hard structured gate remains `40/40`.
- health confirms `moe_repack_sidecar_layers_attached=40`.
- scratch mode confirms `moe_active_scratch_attached=40`.

Speed stays near the safe default. The hot path is therefore not blocked by
manifest lookup or intermediate tensor allocation. The next MoE work should
move directly to grouped native-FP4 gate/up and down math behind the same
sidecar/scratch ABI.

## Artifacts

- `scripts/qwen36_candidate_env_moe_repack_sidecar.env`
- `scripts/qwen36_candidate_env_moe_repack_scratch.env`
- `reports/qwen36_35b/r6000_qwen36_w4a16_moe_repack_sidecar_20260518_164845_moe_repack_runtime_promotion_summary.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_moe_repack_scratch_20260518_165518_moe_repack_scratch_promotion_summary.json`
