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
| RTX PRO 6000 (Blackwell, 2 TB/s) | ~80 t/s estimated | **300 t/s** |

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
