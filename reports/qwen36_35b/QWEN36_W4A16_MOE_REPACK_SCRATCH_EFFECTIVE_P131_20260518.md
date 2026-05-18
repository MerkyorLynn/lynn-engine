# Qwen3.6 35B W4A16 MoE Repack Scratch + Effective-Scale Probe

Date: 2026-05-18

## Candidate

This probe combines the strict MoE repack pieces that were previously tested
separately:

```text
LYNN_MOE_REPACK_SIDECAR_DIR=/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-moe-repack-v0
LYNN_MOE_ACTIVE_SCRATCH=1
LYNN_MOE_EFFECTIVE_SCALE=1
```

The goal was to verify whether the sidecar layout, native-owned scratch, and
memory-neutral effective-scale aliases stack into a promotable serving speedup.

## R6000 Gate Result

| Gate | Result |
|---|---:|
| P37 exact greedy | true |
| P37 median speedup | 0.998x |
| P25 128 decode TPS | 106.90 |
| P25 256 decode TPS | 106.01 |
| P25 512 decode TPS | 107.95 |
| hard structured | 40/40 |
| hard structured mean decode TPS | 107.72 |
| decision | research-only |

## Interpretation

This candidate is quality-safe, but it does not improve the default serving
line.  The combination confirms that allocation, generic manifest lookup, and
runtime global-scale division are not the remaining 122/155 TPS blockers.

P130 remains a useful local active-MoE building block because effective scales
improved the isolated boundary by 1.087x.  However, stacking scratch with that
path flattens the full decode result.  The next useful MoE work must replace
the routed gate/up and down inner math with a true grouped native kernel behind
the same strict active-MoE boundary.

## Artifacts

- `scripts/qwen36_candidate_env_moe_repack_scratch_effective.env`
- `reports/qwen36_35b/r6000_qwen36_w4a16_moe_repack_scratch_effective_20260518_173210_moe_repack_scratch_effective_promotion_summary.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_moe_repack_scratch_effective_20260518_173210_moe_repack_scratch_effective_p25.json`
- `reports/qwen36_35b/r6000_qwen36_w4a16_moe_repack_scratch_effective_20260518_173210_moe_repack_scratch_effective_hard_structured.json`
