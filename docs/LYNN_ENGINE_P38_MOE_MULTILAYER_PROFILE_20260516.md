# Lynn Engine P38 · Multi-layer MoE wall profile

Date: 2026-05-16

## Summary

P38 extends the P37 single-layer MoE segment profile across six sampled 27B
layers: 2, 8, 14, 20, 28, and 36.

The result is intentionally boring in the useful way: there is no isolated slow
layer to tune around. The active MoE wall is uniform across the model.

## Evidence

Report:

```text
reports/p16_155/p38_moe_multilayer_profile.json
```

Mean latency across the six sampled layers:

| Segment | Mean latency |
|---|---:|
| Router top-k | 0.037384 ms |
| Active routed packed NVFP4 experts | 0.112377 ms |
| Shared BF16 expert | 0.059808 ms |
| Current full MoE | 0.193470 ms |

Layer-by-layer current full-MoE latency:

| Layer | Current full MoE |
|---:|---:|
| 2 | 0.193548 ms |
| 8 | 0.193676 ms |
| 14 | 0.193243 ms |
| 20 | 0.193767 ms |
| 28 | 0.193233 ms |
| 36 | 0.193354 ms |

The `current_vs_active_plus_shared_bf16` diff is exact or numerically trivial
across sampled layers (`max_abs` 0 to 6.1e-05, cosine approximately 1.0), so
the segment accounting is measuring the same path as the production MoE call.

## Decision

P38 confirms that the 155 TPS work should not chase layer-specific anomalies.
The next high-leverage work is:

1. Replace the active routed packed NVFP4 expert math (~0.112 ms/layer) with a
   grouped native-FP4 kernel.
2. Then revisit the shared BF16 expert (~0.060 ms/layer), likely with a fused
   or native path that preserves exact generate behavior.
3. Keep router/top-k work lower priority for now (~0.037 ms/layer) because P23
   already ruled out several router-only shortcuts.

## Engineering note

The P38 profiler installs the current best R6000 decode env by default before
constructing `LynnIncrementalRunner`. This avoids repeating the earlier mistake
where a profiler silently fell back to `LYNN_MOE_IMPL=optimized` and skipped
packed NVFP4 aliases.
