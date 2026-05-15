# Lynn Engine P39-P40 · Fast fixed MoE path

Date: 2026-05-16

## Summary

P39 splits the active routed MoE path into inner gate/up and down kernels. P40
then turns the current best R6000 MoE config into a fixed fast path and validates
it with both layer-level and full-generate gates.

This is a small but safe production cleanup. It is not the 155 TPS breakthrough.

## P39 Inner Active-MoE Profile

Report:

```text
reports/p16_155/p39_active_moe_inner_profile.json
```

Mean latency across layers 2, 8, 14, 20, 28, and 36:

| Segment | Mean latency |
|---|---:|
| Router top-k | 0.037632 ms |
| Gate/up packed NVFP4 | 0.033126 ms |
| Down weighted-sum packed NVFP4 | 0.025195 ms |
| Active combined call | 0.120350 ms |
| Shared BF16 expert | 0.061012 ms |

The split gate/up + down result matches the combined active output
(`split_vs_combined` cosine approximately 1.0), so the decomposition is valid.

## P40 Fixed Fast Path

Reports:

```text
reports/p16_155/p40_moe_forward_fast_candidate_gate.json
reports/p16_155/p40_fast_fixed_default_gate.json
```

Layer-level candidate gate:

| Mode | Mean MoE latency |
|---|---:|
| Reference MoE forward | 0.199804 ms |
| Fixed fast candidate | 0.185144 ms |

```text
mean_speedup         = 1.0792x
all_exact_or_trivial = true
```

Full-generate default gate after enabling `LYNN_MOE_FAST_FIXED=1` by default:

```text
new_ids_all_match = true
baseline median   = 100.70 TPS
candidate median  = 100.94 TPS
```

## Decision

Promote the fixed fast path as the default packed-NVFP4 MoE decode path. It
keeps the same current best R6000 kernel config:

```text
gate_block_inter=8
gate_block_hidden=256
down_block_hidden=8
down_block_inter=512
gate_num_warps=4
down_num_warps=8
router_topk_sorted=false
native_active_backend=triton
```

The path remains fail-loud if a user tries to combine it with incompatible
profiling or experimental flags.

## Next

P40 only trims branch/config overhead. P39 shows the bigger target clearly:

- Active routed math split sum is ~0.058 ms/layer.
- Shared BF16 is ~0.061 ms/layer.
- Full production MoE is still ~0.19 ms/layer.

The next major TPS jump needs a fused/grouped native-FP4 active expert kernel,
then a shared-expert pass.
