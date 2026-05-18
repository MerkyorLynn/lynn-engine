# P170 Qwen3.5-9B Dense Gate/Up Phase Profile

Date: 2026-05-19

## Purpose

After P169 admitted `LYNN_DENSE_FFN_GATE_UP_FUSED=1` as an exact opt-in
resident path, P170 fixes the P155 profiler so it can measure the fused
gate/up boundary instead of always timing separate gate and up projections.

Both runs use the safe `linear_graph_only` profile:

```text
LYNN_LINEAR_STATE_UPDATE=inplace
LYNN_LINEAR_BLOCK_GRAPH=1
LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
```

The second run additionally enables:

```text
LYNN_DENSE_FFN_GATE_UP_FUSED=1
```

## R6000 Result

| Phase | Baseline | Fused gate/up | Delta |
|---|---:|---:|---:|
| gate + up projection | 4.9541 ms | 4.7409 ms | -0.2133 ms (-4.30%) |
| dense FFN total | 7.4604 ms | 7.3238 ms | -0.1366 ms (-1.83%) |
| accounted CUDA | 30.9821 ms | 30.0414 ms | -0.9407 ms (-3.04%) |
| wall | 33.1852 ms | 32.2069 ms | -0.9783 ms (-2.95%) |

The fused projection is real and measurable.  It saves roughly 0.21 ms/token in
the gate/up slice, but the total service P150 gain remains small because the
dominant blocks are still linear-attention/SSM and full-attention work, plus
their graph/capture boundary behavior.

## Decision

Keep `LYNN_DENSE_FFN_GATE_UP_FUSED=1` as an exact opt-in building block.  The
next 9B speed step should be a larger exact boundary: either fuse activation
plus down into the dense FFN island, or move back to the larger linear/SSM
boundary where P170 still shows the largest absolute time.

## Artifacts

- `benchmarks/p155_qwen35_9b_dense_ffn_phase_profile.py`
- `reports/qwen35_9b/p155_qwen35_9b_dense_ffn_phase_profile_20260519_0840_lineargraph_p155v2.json`
- `reports/qwen35_9b/p155_qwen35_9b_dense_ffn_phase_profile_20260519_0840_lineargraph_densegateup_p155v2.json`
