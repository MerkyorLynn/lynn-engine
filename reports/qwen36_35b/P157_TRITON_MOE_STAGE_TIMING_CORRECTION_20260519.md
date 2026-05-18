# Qwen3.6 35B W4A16 P157 Triton MoE Stage Timing Correction

Date: 2026-05-19

## Purpose

P147 produced correct Triton-stage references, but its `gateup_ms` timing called
a helper that computed both gate/up and down before returning the intermediate.
This made the stage timing look larger than it really is. P157 measures gate/up,
down, and combined exact Triton active-MoE stage timings separately.

## Result

Artifact:
`reports/qwen36_35b/p157_triton_moe_stage_timing_20260519_042733.json`

| Check | Result |
|---|---:|
| inter exact vs P147 | 18/18 |
| out exact vs P147 | 18/18 |
| gate/up mean | 0.03005 ms |
| down mean | 0.02479 ms |
| gate + down mean | 0.05484 ms |
| combined sequential mean | 0.05228 ms |

## Correction

The earlier interpretation that native packed MoE was faster than Triton was
wrong. P152's native packed candidate was about 0.090 ms, while exact Triton
stage is about 0.052 ms combined. Native packed is currently both non-exact and
slower at fixture stage.

## Updated Direction

Do not prioritize native packed-MoE replacement as the immediate 35B speed path.
The exact Triton math is already the faster stage implementation. The next MoE
work should keep the Triton contract and reduce boundary overhead around it:

1. caller-owned / graph-owned Triton stage buffers;
2. launch-boundary coarsening without changing math;
3. resident graph reuse around the exact Triton active-MoE stage.

Native C++ should remain a research branch unless it can first become exact and
beat the corrected 0.052 ms Triton stage.
