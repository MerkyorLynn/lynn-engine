# Qwen3.5-9B NVFP4 Runtime Progress 2026-05-19

## Current Route

Qwen3.5-9B dense is now the primary fast-serving branch. 35B A3B remains the
quality/MoE side branch.

The current safe Lynn-native W4A16 NVFP4 candidate is:

```bash
source scripts/qwen35_9b_candidate_env_convstrict.env
```

It keeps the exact `linear_graph_only` path and adds only
`LYNN_LINEAR_ATTN_CONV_BACKEND=triton_torch_silu`.

## R6000 Results

| Candidate | Gate | Result |
|---|---:|---|
| `linear_graph_only` reference | P183 direct | 59.02 decode TPS |
| `graph_plus_conv_triton` | P183 direct | 62.12 decode TPS, best exact candidate |
| `graph_plus_conv_triton` | P184 70 hard prompts | 70/70 exact, 1.031x mean speedup |
| `graph_plus_conv_triton` | P150 OpenAI service | 128: 61.32, 256: 62.25, 512: 62.09 decode TPS |

This is a modest but clean default-speed gain over the prior 9B strict line.

## AMBER Line

`fast_no_packed_decode` is still the fastest observed Lynn-native 9B service
profile:

| Candidate | P150 OpenAI service |
|---|---:|
| `fast_no_packed_decode` | 128: 75.95, 256: 77.00, 512: 77.03 decode TPS |

It is not promotable yet. P149/P183 show exact-greedy drift in the component
sweep. Keep it as AMBER/research until the drift source is eliminated.

Known drift contributors from P183:

| Knob group | Exactness vs strict graph reference | Decode TPS |
|---|---:|---:|
| recurrent prepare + GQA | 2/3 | 66.18 |
| QK/RoPE triton pair | 2/3 | 64.41 |
| fast no packed, no native LM | 2/3 | 73.11 |
| fast no packed, native LM | 1/3 | 75.69 |

## Quality Tracking

Thinking-off 9B quality snapshot:

| Quant | MMLU | GPQA Diamond |
|---|---:|---:|
| BF16 | 77.20% | 44.95% |
| Q4_K_M | 76.00% | 37.37% |
| Lynn W4A16 NVFP4 | 75.20% | 42.93% |

Thinking-on 32K reruns are in progress. The evaluator parser was fixed on
2026-05-19 to prefer the last explicit `Answer: X`, because the earlier parser
could pick option letters from the reasoning text.

## Next Work

1. Make `fast_no_packed_decode` exact, or recover most of its speed with an
   exact subset of recurrent/QK/native-LM changes.
2. Run the CUDA `llama.cpp` Q4_K_M 32K baseline on R6000. A CPU-only
   `llama-server` was accidentally used for the first 32K run; use
   `/root/autodl-tmp/llama.cpp/build-cuda/bin/llama-server`.
3. Re-run 9B BF16/Q4_K_M/NVFP4 MMLU and GPQA in thinking-on 32K mode for a
   clean release matrix.
