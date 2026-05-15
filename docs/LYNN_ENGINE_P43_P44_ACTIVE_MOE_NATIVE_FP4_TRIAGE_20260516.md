# Lynn Engine P43-P44 · Active MoE Native FP4 Triage

Date: 2026-05-16

## Summary

P43-P44 closes three tempting but insufficient routes toward the 155 TPS target:

1. Shared BF16 expert micro-optimization.
2. Merged-top-k Triton gate/up scheduling.
3. Cross-expert `torch._scaled_mm` composition.

All three are useful evidence, but none is the breakthrough.  The remaining
path is a real grouped/block-diagonal active expert FP4 kernel.

## P43 Shared Expert Inner Profile

Report:

```text
reports/p16_155/p43_shared_expert_inner_profile.json
```

Mean across layers 2, 8, 14, 20, 28, and 36:

| Segment | Mean latency |
|---|---:|
| Separate shared expert | 0.060850 ms |
| Fused gate/up shared expert | 0.055598 ms |
| Fused gate/up projection only | 0.009093 ms |
| Down projection only | 0.009147 ms |
| Shared gate only | 0.016538 ms |

```text
fused_speedup_vs_separate_mean = 1.094x
fused_vs_separate cosine       ~= 1.0
```

Decision: keep the existing fused shared gate/up path, but do not spend the next
155-TPS sprint on shared BF16.  The absolute saving is only about 0.005 ms per
sampled layer.

## P44-A Merged-Top-K Gate/Up Retest

Report:

```text
reports/p16_155/p44_p26_gateup_merged_topk_layers.json
```

This retests the earlier P26 idea: launch one program per intermediate block and
loop over all top-k experts inside the program.  It is numerically aligned but
slow.

Representative result:

```text
reference Triton gate/up ~= 0.033 ms
best merged-top-k       ~= 0.068-0.069 ms
speedup_vs_reference    ~= 0.48x
```

Decision: do not promote merged-top-k scheduling.  Reducing program count loses
too much parallelism for this shape.

## P44-B Cross-Expert `_scaled_mm` Probe

Report:

```text
reports/p16_155/p44_gateup_cross_expert_scaled_mm_probe.json
```

Shape:

```text
activation rows: [8, 2048]
selected packed weight: [8192, 1024]
dense output: [8, 8192]
kept diagonal blocks: [8, 1024]
wasted cross-expert factor: 8
```

Mean across sampled layers:

| Path | Mean latency |
|---|---:|
| Current Triton gate/up | 0.033207 ms |
| Native FP4 cross-expert mm-only | 0.084831 ms |
| Native FP4 cross-expert quant+mm | 0.225101 ms |

```text
native_mm_only_speedup_vs_reference       = 0.391x
native_quant_plus_mm_speedup_vs_reference = 0.148x
min cosine vs Triton reference            = 0.97698
```

Decision: composing `torch._scaled_mm` is not a production route.  It both
over-computes top-k cross terms and introduces activation-FP4 drift.  It is
valuable as a negative result because it rules out the easy shortcut.

## What This Means For 155 TPS

The current stable/default R6000 path is around 100.8 TPS after P40.  The safe
small wins are now mostly exhausted:

- shared expert is already small;
- merged-top-k scheduling is slower;
- `torch._scaled_mm` composition is slower and drifts;
- `cuda_scalar` remains diagnostic only after P42.

The next real breakthrough must replace the active expert gate/up and down
inner loops with a Lynn-owned grouped/block-diagonal FP4 kernel that:

- consumes packed E2M1 weights directly;
- avoids cross-expert over-compute;
- keeps router/top-k order and routing weights exact enough for greedy parity;
- fuses enough work to beat the current Triton pair;
- has fail-loud gates before any production promotion.

## Next: P45 Kernel Contract

P45 should stop probing wrappers and define the actual kernel boundary:

```text
inputs:
  hidden[2048] bf16
  expert_ids[top_k=8] int32
  routing_weights[top_k=8] fp32
  gate_up_packed[top_k, 1024, 1024] uint8 view/gather
  gate_up_scale[top_k, 1024, 128] fp32
  down_packed[top_k, 2048, 256] uint8 view/gather
  down_scale[top_k, 2048, 32] fp32
outputs:
  moe_out[2048] bf16/fp16
```

The first milestone can be a fused active-MoE CUDA extension that is exact-ish
against the current Triton reference on one layer.  Production promotion still
requires multi-layer and full-generate parity gates.
