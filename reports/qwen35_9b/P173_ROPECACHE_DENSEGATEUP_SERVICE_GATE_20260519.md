# P173 Qwen3.5-9B RoPE-Cache + Dense Gate/Up Service Gate

Date: 2026-05-19

## Purpose

P169 made `LYNN_DENSE_FFN_GATE_UP_FUSED=1` an exact opt-in building block, but
the OpenAI service gate was nearly flat at 512 tokens. P173 stacks the existing
full-attention RoPE cache on top of that exact dense gate/up path:

```text
LYNN_DENSE_FFN_GATE_UP_FUSED=1
LYNN_FULL_ATTN_ROPE_CACHE=1
LYNN_FULL_ATTN_ROPE_CACHE_MAX_SEQ=65536
```

The base serving script still applies the safe 9B linear graph profile:

```text
LYNN_LINEAR_STATE_UPDATE=inplace
LYNN_LINEAR_BLOCK_GRAPH=1
LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
```

## P25 Result

| Max tokens | Decode TPS mean |
|---:|---:|
| 128 | 61.85 |
| 256 | 62.45 |
| 512 | 62.55 |

All 9 P25 requests reported `linear_block_graph_reused=true`.

## Readout

This is a small but real service-side improvement over the previous safe 9B
profile:

| Profile | 512 decode TPS |
|---|---:|
| Linear graph only | 61.69 |
| Dense gate/up fused | 61.85 |
| Dense gate/up fused + RoPE cache | 62.55 |

The gain is not large enough to call a new 9B default by itself, but it is a
clean opt-in candidate for the next 9B release gate. The next useful 9B work is
still a larger dense FFN boundary or repacked TensorCore path; this probe only
removes avoidable full-attention RoPE setup overhead.

## Artifacts

- `scripts/qwen35_9b_candidate_env_ropecache_densegateup.env`
- `reports/qwen35_9b/p150_qwen35_9b_nvfp4_linear_graph_p25_20260519_1145_ropecache.json`
- `reports/qwen35_9b/p150_qwen35_9b_nvfp4_linear_graph_summary_20260519_1145_ropecache.json`
