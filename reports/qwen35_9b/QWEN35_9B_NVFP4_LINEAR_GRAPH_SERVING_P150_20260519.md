# Qwen3.5-9B NVFP4 Linear-Graph Serving Gate

**Date:** 2026-05-19  
**Model:** `/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0`  
**Gate:** OpenAI-compatible P25 decode TPS

## Verdict

`linear_graph_only` is **P25_READY** for the 9B NVFP4 serving path.

The server was launched with only:

```text
LYNN_LINEAR_STATE_UPDATE=inplace
LYNN_LINEAR_BLOCK_GRAPH=1
LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
```

No packed decode, native in-proj, native LM head, or broader 35B fast-profile
knobs were enabled.

## P25 Result

| Max tokens | Decode TPS mean |
|---:|---:|
| 128 | 60.80 |
| 256 | 61.47 |
| 512 | 61.69 |

All 9 P25 requests reported `linear_block_graph_reused=true`.

## Readout

This upgrades the safe R6000 9B NVFP4 service line from the old 40.9 TPS
matrix result to about 61.7 decode TPS at 512 tokens. It does not close the
llama.cpp Q4_K_M gap: Q4_K_M remains the speed reference at 168.23 TPS single
and 420.63 TPS x8. The useful conclusion is narrower but solid: 9B NVFP4 has a
safe graph-only serving profile, and further speed work should focus on dense
FFN packed/fused kernels rather than broad fast-profile toggles.

## Artifacts

- `scripts/r6000_qwen35_9b_nvfp4_linear_graph_serving_gate.sh`
- `reports/qwen35_9b/p150_qwen35_9b_nvfp4_linear_graph_p25_20260519_0406.json`
- `reports/qwen35_9b/p150_qwen35_9b_nvfp4_linear_graph_summary_20260519_0406.json`
