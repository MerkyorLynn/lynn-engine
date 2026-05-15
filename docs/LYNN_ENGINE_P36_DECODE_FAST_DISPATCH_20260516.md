# Lynn Engine P36 Decode Fast Dispatch Gate

Date: 2026-05-16

## Summary

P36 hoists decode-time dispatch choices out of the per-layer hot loop:

- MoE decode implementation is resolved once in `LynnIncrementalRunner`.
- Linear-attention recurrent backend is fixed once per runner.
- Linear state update policy is fixed once per runner.

This is deliberately a zero-math-change cleanup. It does not alter kernels,
routing, weights, graph capture, or logits.

## R6000 Result

Report:

```text
reports/p16_155/p36_decode_fast_dispatch_gate.json
```

| Mode | Median Decode TPS | Mean Decode TPS | Token IDs |
|---|---:|---:|---|
| Legacy env/import dispatch | 100.55 | 100.29 | reference |
| Runner-fixed dispatch | 100.53 | 100.54 | exact match |

```text
new_ids_all_match = true
median_speedup    = 0.9998x
promote_default   = true
```

## Decision

Keep runner-fixed dispatch enabled by default because it is exact and removes
unnecessary Python work from the hot path. Do **not** count P36 as a TPS
breakthrough: the benchmark shows the current 100 -> 155 TPS gap is not caused
by MoE function import/env dispatch.

The remaining blocker is still active routed expert compute and/or a stricter
graph-owned full-token serving path, not simple Python dispatch overhead.

## Next

Continue P37 toward the active-MoE kernel:

- Keep P32-P34 `cuda_scalar` generate-path guard in place.
- Do not promote unsafe 121.7 TPS `cuda_scalar + graph` results.
- Focus on grouped native-FP4 active expert kernels or an equivalent exact
  tensor-core path.
