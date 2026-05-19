# R6000 Qwen3.5-9B Q4_K_M-imatrix CUDA Baseline

**Date:** 20260519
**Status:** 🟢 DONE
**Engine:** llama.cpp CUDA (CUDA 12.8, CMAKE_CUDA_ARCHITECTURES=120, flash-attn)
**Model:** `/root/autodl-tmp/models/Qwen3.5-9B-Q4_K_M.gguf` (5.49 GiB)
**Binary:** `/root/autodl-tmp/llama.cpp/build-cuda/bin/llama-server` (rev `856c3ad`)
**GPU layers:** 99 | **Context:** 32768

## Single-Stream Decode TPS

| max_tokens | Prompt tokens | Completion tokens | Wall TPS | Elapsed (s) |
|---:|---:|---:|---:|---:|
| 128 | 28 | 128 | **126.0** | 1.016 |
| 256 | 28 | 256 | **159.6** | 1.604 |
| 512 | 28 | 512 | **164.5** | 3.112 |

## Concurrent Decode TPS (total)

| Concurrency | Total completion tokens | Batch wall TPS | Elapsed (s) |
|---:|---:|---:|---:|
| 2 | 512 | **163.3** | 3.135 |
| 4 | 1024 | **165.1** | 6.203 |
| 8 | 2048 | **166.1** | 12.329 |

## Long-Context Prefill + Decode

| Chars | Prompt tokens | Completion tokens | Wall TPS | Elapsed (s) | Status |
|---:|---:|---:|---:|---:|:---:|
| 32,768 | 7,156 | 64 | **55.5** | 1.153 | ✅ |

## Cross-Reference: Lynn NVFP4 Watcher (R6000)

| Metric | Lynn NVFP4 Watcher | Typical | This CUDA Baseline | Delta |
|--------|-------------------|---------|-------------------|-------|
| Single 512 TPS | single_tps (P25 probe) | ~104-107 TPS | **164.5** | +59.0 |
| Concurrent 8 total | concurrent_total_tps | ~380-400 TPS | **166.1** | -223.9 |
| GPQA accuracy | gpqa_score | ~50.0% | **—** | +0.0pp |

