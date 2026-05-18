# P175 Qwen3.5-9B Act-Scratch Stacked Service Gate

Date: 2026-05-19

## Verdict

`LYNN_NATIVE_FP4_ACT_SCRATCH=1` does not move the 9B NVFP4 service line.

Stack tested on R6000:

- `LYNN_DENSE_FFN_GATE_UP_FUSED=1`
- `LYNN_FULL_ATTN_ROPE_CACHE=1`
- `LYNN_FULL_ATTN_ROPE_CACHE_MAX_SEQ=65536`
- `LYNN_NATIVE_FP4_ACT_SCRATCH=1`
- linear block graph reuse/prewarm

| Max Tokens | Decode TPS |
|---:|---:|
| 128 | 61.82 |
| 256 | 62.42 |
| 512 | 62.52 |

This is effectively flat versus P173's 62.55 TPS at 512 tokens. Keep the 9B
NVFP4 release posture unchanged: P150/P173 are the current Lynn-native path,
while Q4_K_M llama.cpp remains the portable speed path.

## Artifacts

- `reports/qwen35_9b/p150_qwen35_9b_nvfp4_linear_graph_p25_20260519_0830_actscratch_ropecache_densegateup.json`
- `reports/qwen35_9b/p150_qwen35_9b_nvfp4_linear_graph_summary_20260519_0830_actscratch_ropecache_densegateup.json`
