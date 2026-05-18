# P168 Qwen3.6-35B Linear/GDN Segment Census

Date: 2026-05-19

## Verdict

P168 confirms the next worthwhile exact-boundary work is the linear/GDN core,
not another router/shared-expert micro knob.

Across all 30 linear-attention layers, the repeated measured cost is:

| Segment | Sum Across 30 Layers | Mean / Layer | Share of Segmented Core |
|---|---:|---:|---:|
| fused native FP4 in-proj | 2.107 ms/token | 0.0702 ms | 38.4% |
| recurrent fused prepare / GDN | 1.132 ms/token | 0.0377 ms | 20.6% |
| conv update | 0.984 ms/token | 0.0328 ms | 18.0% |
| gated RMSNorm | 0.605 ms/token | 0.0202 ms | 11.0% |
| out proj BF16 | 0.449 ms/token | 0.0150 ms | 8.2% |
| split qkv/repeat | 0.206 ms/token | 0.0069 ms | 3.8% |

The recomposed full linear core averages 0.2998 ms/layer, or 8.993 ms across
30 layers under the measurement harness.

## Implication

The top three pieces, in-proj + recurrent/GDN + conv, account for about
4.224 ms/token of repeatable work. A useful 35B jump now needs a larger exact
boundary around these pieces, or an offline layout/repack that reduces the
in-proj cost. The small surfaces already tested are exhausted:

- P164 router softmax scratch was exact but flat.
- P165 prepared Triton active-MoE boundary was exact but below default.
- P166 block-shape sweep found only the current Triton active-MoE config exact.
- P167 shared-expert prepared path was exact but too small to move service TPS.

## Next Work

1. Build a fixture-style linear-core contract that stores post-input-norm hidden,
   conv/recurrent state slices, and expected in-proj/conv/recurrent/norm/out
   outputs for representative linear layers.
2. Probe an exact caller-owned scratch boundary for `in_proj -> conv -> recurrent`
   first. It targets the largest 77% of the segmented core without touching MoE.
3. Keep default serving unchanged until the boundary passes local exactness, P37,
   P25, and hard structured gates.

## Artifacts

- `reports/qwen36_35b/p168_qwen36_linear_core_segment_census_20260519_0729.json`
- `benchmarks/p168_qwen36_linear_core_segment_census.py`
- `scripts/r6000_qwen36_linear_core_segment_census.sh`
