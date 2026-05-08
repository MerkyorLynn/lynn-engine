# Lynn Engine

> Custom inference engine for **Qwen 3.6 35B-A3B** on NVIDIA Blackwell (DGX Spark + RTX PRO 6000).

A purpose-built single-model inference engine. Inspired by [antirez/ds4](https://github.com/antirez/ds4) (DeepSeek V4 Flash on Apple Metal). Goal: **2-3x throughput vs vLLM/SGLang** on the same hardware by going vertical with custom training (LoRA + expert pruning) + asymmetric mixed-precision quantization + disk-backed KV cache with SHA1 prefix matching.

## Status

📐 **Design draft** — see [docs/DESIGN.md](docs/DESIGN.md)
🔬 **Phase 0 (baseline) → Phase 1 (Triton spike)** pending Step-3.5 GGUF download + LoRA training completion on DGX Spark.

## Performance targets

| Hardware | Today (vLLM/SGLang) | Lynn Engine target |
|---|---|---|
| DGX Spark (GB10, 273 GB/s) | 60-70 t/s | **120 t/s** |
| RTX PRO 6000 (Blackwell, 96GB GDDR7, 2 TB/s) | **200+ t/s** (5090 TP=2 实测 184 单流 / 248 5 并发,RTX PRO 6000 单卡带宽更高) | **300-500 t/s** (Phase 4 asymmetric + MTP) |

Lynn Engine targets are derived from memory-bandwidth ceilings:
  - FP8 (3 GB active per token forward): 2000 GB/s ÷ 3 GB = ~660 t/s theoretical max single batch
  - NVFP4+FP8 mixed (2 GB per token): ~1000 t/s theoretical max
  - vLLM today realizes ~30% of bandwidth ceiling. Phase 4 target = 50-75%.

## Why this exists

Lynn brain serves Qwen 3.6 35B-A3B as the primary route for thousands of agent requests/day. Generic engines (vLLM, SGLang) realize ~40% of theoretical memory-bandwidth ceiling. Single-model lock-in + Blackwell sm_12x specialization unlocks the rest.

## Repository structure

See [docs/DESIGN.md §11 Repository Layout](docs/DESIGN.md#11-repository-layout).

## Trade-offs (honest)

- ❌ Locked to Qwen 3.6 35B-A3B + Blackwell sm_12x.
- ❌ If model lineage replaced incompatibly, 4-6 weeks rewrite.
- ✅ Fits Lynn brain's deployment exactly.
- ✅ Vertical with LoRA + pruning training pipeline already in flight.

## License

TBD (likely MIT, decided before Phase 6 production cutover).
