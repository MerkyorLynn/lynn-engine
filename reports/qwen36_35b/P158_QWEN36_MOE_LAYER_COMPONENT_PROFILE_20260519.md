# Qwen3.6 35B W4A16 P158 MoE Layer Component Profile

Date: 2026-05-19

## Purpose

After P157 corrected active top-k Triton timing, P158 profiles the real MoE
sub-layer around that exact active stage using real 35B layer weights plus p138
slot-packed fixtures.

## Result

Artifact:
`reports/qwen36_35b/p158_qwen36_moe_layer_component_profile_20260519_043530.json`

| Component | Mean ms/layer |
|---|---:|
| router + top-k + softmax | 0.04428 |
| active gate/up | 0.03279 |
| active down | 0.02980 |
| active combined | 0.05907 |
| shared expert | 0.03558 |
| shared gate/add finalize | 0.03121 |
| total MoE sub-layer | 0.19578 |

## Interpretation

Active top-k Triton is not the dominant standalone cost. The full MoE sub-layer
is a collection of medium-sized boundaries:

- router/top-k/softmax is comparable to shared expert;
- active exact Triton is about 30% of total MoE sub-layer time;
- finalize/shared gate-add is also material at ~0.031 ms/layer.

Across 30 linear-attention layers, a 0.196 ms MoE sub-layer implies about
5.87 ms/token before attention and other layer work. Meaningful gains must
coarsen or fuse several exact boundaries; replacing only the active stage is
unlikely to move the full server enough.

## Updated Next Targets

1. Router/top-k path: reduce Python/Torch boundary around router + top-k +
   softmax, keeping unsorted top-k semantics.
2. Shared expert finalize: fuse shared gate/add where exact; prior AMBER
   shared-gate variants need exact-greedy caution.
3. Active Triton stage: preserve exact Triton math and focus on graph/reuse or
   caller-owned buffer integration, not native replacement.
