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
| 128 | 28 | 128 | **131.6** | 0.973 |
| 256 | 28 | 256 | **160.2** | 1.598 |
| 512 | 28 | 512 | **165.8** | 3.089 |

## Concurrent Decode TPS (total)

| Concurrency | Total completion tokens | Batch wall TPS | Elapsed (s) |
|---:|---:|---:|---:|
| 2 | 512 | **270.6** | 1.892 |
| 4 | 1024 | **330.8** | 3.096 |
| 8 | 2048 | **413.5** | 4.953 |

## Long-Context Prefill + Decode

| Chars | Prompt tokens | Completion tokens | Wall TPS | Elapsed (s) | Status |
|---:|---:|---:|---:|---:|:---:|
| 4,096 | 3,704 | 256 | **173.8** | 1.473 | ✅ |
| 16,384 | 14,384 | 256 | **114.4** | 2.238 | ✅ |
| 32,768 | — | — | — | — | ❌ failed |

Note: failures at the largest context are likely affected by `--parallel 8` slot partitioning; rerun with `PARALLEL=1` for true single-request long-context capacity.

## Cross-Reference: Lynn NVFP4 Watcher (R6000)

| Metric | Lynn NVFP4 Watcher | Typical | This CUDA Baseline | Delta |
|--------|-------------------|---------|-------------------|-------|
| Single 512 TPS | single_tps (P25 probe) | ~104-107 TPS | **165.8** | +60.2 |
| Concurrent 8 total | concurrent_total_tps | ~380-400 TPS | **413.5** | +23.5 |
| GPQA accuracy | gpqa_score | ~50.0% | **—** | +0.0pp |

