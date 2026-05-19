# Qwen3.5-9B Q4_K_M-imatrix llama.cpp CUDA Baseline

**Date:** 2026-05-19
**Branch:** `qwen/qwen35-9b-q4km-cuda-baseline-20260519`
**Status:** 🟢 READY (harness complete; execution requires GGUF on R6000)

## Purpose

Establish a clean R6000 llama.cpp **CUDA** baseline for Qwen3.5-9B Q4_K_M-imatrix.

**⚠️ CPU-only baseline is invalid.** The earlier 32K thinking GPQA run used
`/root/autodl-tmp/llama.cpp/build/bin/llama-server` (CPU-only build). That binary
accepts `--n-gpu-layers` but does NOT link `ggml-cuda` — all numbers from that
run are meaningless for GPU comparison.

This baseline uses `/root/autodl-tmp/llama.cpp/build-cuda/bin/llama-server`,
built with CUDA 12.8 and `CMAKE_CUDA_ARCHITECTURES=120`.

## Release Tracks

| Track | Target HW | Model | Engine |
|-------|-----------|-------|--------|
| Lynn NVFP4 | NVIDIA R6000 | Qwen3.5/3.6-9B | Lynn Engine (CUDA) |
| Mac Q4_K_M | Apple Silicon | Qwen3.5-9B | llama.cpp |
| **R6000 Q4_K_M CUDA** | **NVIDIA R6000** | **Qwen3.5-9B** | **llama.cpp CUDA (this baseline)** |
| ~~R6000 Q4_K_M CPU~~ | ~~NVIDIA R6000~~ | ~~Qwen3.5-9B~~ | ~~llama.cpp CPU~~ (invalid) |

## What It Measures

1. **Single-stream decode TPS** at 128/256/512 max_tokens
2. **Concurrent total TPS** at 2/4/8 parallel requests
3. **Long-context prefill+decode** smoke at 4k/16k/32k characters
4. **GPQA Diamond 32K thinking** accuracy via OpenAI-compatible API

## Files

| File | Purpose |
|------|---------|
| `scripts/r6000_qwen35_9b_q4km_cuda_baseline.sh` | Main harness (CUDA binary enforcement, server, bench + GPQA) |
| `scripts/summarize_qwen35_9b_q4km_cuda_baseline.py` | JSON → Markdown summary generator |
| `scripts/openai_gpqa_diamond_eval.py` | GPQA Diamond evaluator (existing) |
| `reports/qwen35_9b/` | Report output directory |

## Quick Start (R6000)

```bash
# 0. Verify CUDA binary exists
ls -la /root/autodl-tmp/llama.cpp/build-cuda/bin/llama-server

# 1. Verify GGUF on disk (prefer imatrix variant)
ls -lh /root/autodl-tmp/models/Qwen3.5-9B-Q4_K_M-imatrix.gguf \
       /root/autodl-tmp/models/Qwen3.5-9B-Q4_K_M.gguf

# 2. Run full baseline (perf + GPQA)
bash scripts/r6000_qwen35_9b_q4km_cuda_baseline.sh

# 3. Or run subsets
PERF=1 bash scripts/r6000_qwen35_9b_q4km_cuda_baseline.sh   # throughput only
GPQA=1 bash scripts/r6000_qwen35_9b_q4km_cuda_baseline.sh   # GPQA only

# 4. Results
ls -la reports/qwen35_9b/*cuda*baseline*.json
python3 scripts/summarize_qwen35_9b_q4km_cuda_baseline.py --latest
```

## CUDA Binary Build (if missing)

```bash
cd /root/autodl-tmp/llama.cpp
cmake -B build-cuda \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j$(nproc)
ls -la build-cuda/bin/llama-server
```

## GGUF Download (if missing)

```bash
# HuggingFace CLI — imatrix variant (preferred)
huggingface-cli download Qwen/Qwen3.5-9B-GGUF qwen3.5-9b-q4_k_m-imatrix.gguf \
  --local-dir /root/autodl-tmp/models/

# HuggingFace CLI — standard Q4_K_M
huggingface-cli download Qwen/Qwen3.5-9B-GGUF qwen3.5-9b-q4_k_m.gguf \
  --local-dir /root/autodl-tmp/models/

# ModelScope (China mirror)
modelscope download Qwen/Qwen3.5-9B-GGUF qwen3.5-9b-q4_k_m.gguf \
  --local_dir /root/autodl-tmp/models/

# Direct URL
export GGUF_URL="https://huggingface.co/Qwen/Qwen3.5-9B-GGUF/resolve/main/qwen3.5-9b-q4_k_m-imatrix.gguf"
bash scripts/r6000_qwen35_9b_q4km_cuda_baseline.sh
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ROOT` | `/root/autodl-tmp/models` | Model search root |
| `GGUF` | (auto-discovered) | Explicit GGUF path |
| `LLAMA_SERVER` | (auto-discovered) | Override CUDA binary path |
| `HF_REPO` | (none) | HuggingFace repo ID for download |
| `MS_REPO` | (none) | ModelScope repo ID for download |
| `GGUF_URL` | (none) | Direct download URL |
| `REPORT_ROOT` | `/root/autodl-tmp/reports/qwen35_9b` | Report output dir |
| `PORT` | `18197` | llama-server port |
| `HOST` | `127.0.0.1` | Server bind address |
| `CTX_SIZE` | `32768` | Context window size |
| `PARALLEL` | `8` | Server parallel slots |
| `N_GPU_LAYERS` | `99` | GPU offload layers (99 = full) |
| `SINGLE_MAX_TOKENS` | `128 256 512` | Single-stream decode lengths |
| `CONCURRENCY` | `2 4 8` | Concurrent request counts |
| `LONG_CONTEXT_CHARS` | `4096 16384 32768` | Long-context char counts |
| `GPQA` | `0` | Set to 1 for GPQA-only run |
| `PERF` | `0` | Set to 1 for perf-only run |
| `GPQA_SEED` | `42` | GPQA random seed |

## CUDA Binary Discovery (strict)

The script uses **CUDA binary only** — no CPU fallback:

1. `$LLAMA_SERVER` env var (if set)
2. `/root/autodl-tmp/llama.cpp/build-cuda/bin/llama-server`

If neither is found → **hard error** with build instructions. No silent fallback
to CPU-only build.

## GGUF Auto-Discovery Order

1. `$GGUF` env var (if set and file exists)
2. `$MODEL_ROOT/Qwen3.5-9B-Q4_K_M-imatrix.gguf` (imatrix preferred)
3. `$MODEL_ROOT/Qwen3.5-9B-Q4_K_M.gguf` (standard fallback)
4. Glob: `$MODEL_ROOT/*Qwen3.5*9B*Q4*K*M*.gguf`

## CUDA Server Arguments

```
--n-gpu-layers 99      # full GPU offload
--flash-attn           # flash attention
--cont-batching        # continuous batching for concurrent requests
```

The script also verifies the binary links `ggml-cuda` (via `strings` check).

## JSON Report Schema

```json
{
  "schema": "lynn-qwen35-9b-q4km-cuda-baseline-v1",
  "status": "DONE | PENDING_DOWNLOAD",
  "model_id": "Qwen3.5-9B",
  "quant": "Q4_K_M-imatrix",
  "engine": "llama.cpp CUDA",
  "engine_detail": "CUDA 12.8, CMAKE_CUDA_ARCHITECTURES=120, flash-attn",
  "model_path": "/path/to/gguf",
  "size_gib": "5.20",
  "llama_cpp_binary": "/path/to/llama-server",
  "git_rev": "abc1234",
  "n_gpu_layers": 99,
  "ctx_size": 32768,
  "parallel": 8,
  "single_tps": { ... },
  "concurrent_tps": { ... },
  "long_context": { ... },
  "errors": []
}
```

## Cross-Reference: Lynn NVFP4 Watcher Fields

| CUDA Baseline Field | Lynn NVFP4 Watcher Field | Expected CUDA vs Lynn |
|---------------------|--------------------------|----------------------|
| `single_tps.512.wall_tps` | `single_tps` (P25 probe) | CUDA ~2x faster |
| `concurrent_tps.8.batch_wall_tps` | `concurrent_total_tps` | CUDA ~1.3x faster |
| GPQA accuracy | `gpqa_score` | Should match (~50%) |

## Background: Why CPU Baseline Is Invalid

The CPU-only `llama-server` (`build/bin/llama-server`) silently accepts
`--n-gpu-layers 99` but runs all computation on CPU. Evidence:

- Server log shows no CUDA memory allocation
- Decode TPS matches CPU-only expectations (~15-20 TPS for 9B)
- `strings build/bin/llama-server | grep ggml-cuda` returns nothing

The CUDA build (`build-cuda/bin/llama-server`) properly links `ggml-cuda` and
offloads matrix multiplication to the GPU.

## Expected Performance (R6000 RTX PRO 6000 Blackwell)

Based on 35B Q4_K_M CUDA baseline (207 TPS single, 501 TPS concurrent 8),
the 9B model should be significantly faster. Estimates:

| Metric | 35B Q4_K_M CUDA | 9B Q4_K_M CUDA (expected) |
|--------|-----------------|---------------------------|
| Single 512 TPS | 207 | 400+ |
| Concurrent 8 total | 501 | 800+ |
| GPQA accuracy | 50.0% | ~37-50% |
