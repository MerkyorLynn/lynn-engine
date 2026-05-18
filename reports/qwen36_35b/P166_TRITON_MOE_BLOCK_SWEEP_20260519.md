# P166 Qwen3.6-35B Triton Active-MoE Block Sweep

Date: 2026-05-19

## Purpose

After P165 closed the prepared-wrapper boundary, P166 checked whether the
current exact Triton active-MoE authority had any safe block-shape headroom.
The sweep used P157's Triton-stage fixtures and varied gate/up and down block
parameters without changing the mathematical kernels.

## Result

| Metric | Value |
|---|---:|
| Configs tested | 40 |
| Exact configs | 1 |
| Exact config | gate 8x256, down 8x512, warps 4/8 |
| Exact config combined mean | 0.05348 ms |
| Exact config gate+down mean | 0.05761 ms |

The only 18/18 exact configuration is the current default.

## Fastest Near Misses

| Config | Inter exact | Out exact | Combined ms | Gate+down ms |
|---|---:|---:|---:|---:|
| gate 8x256, down 8x512, warps 4/16 | 18/18 | 13/18 | 0.05341 | 0.05795 |
| gate 8x256, down 8x256, warps 4/8 | 18/18 | 11/18 | 0.05404 | 0.05749 |
| gate 8x256, down 8x512, warps 4/4 | 18/18 | 14/18 | 0.05413 | 0.06250 |
| gate 8x128, down 8x512, warps 4/8 | 6/18 | 14/18 | 0.05495 | 0.06004 |

These are not promotable. The small timing differences are below service-level
noise and lose exactness.

## Decision

Close Triton block-shape tuning for the current active-MoE authority. The exact
default shape is already the safe point. Further 35B progress has to come from
a larger boundary or offline layout change that removes real launches or memory
traffic, not from changing block sizes.

## Artifacts

- `reports/qwen36_35b/p166_triton_moe_block_sweep_summary.json`
