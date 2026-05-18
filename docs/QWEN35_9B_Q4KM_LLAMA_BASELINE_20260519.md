# Qwen3.5-9B Q4_K_M llama.cpp Baseline Harness

**Date:** 2026-05-19
**Branch:** `mimo/qwen35-9b-q4km-llamacpp-baseline-20260519`
**Status:** 🟢 READY (harness complete; execution requires GGUF on R6000)

## Purpose

Establish Qwen3.5-9B Q4_K_M llama.cpp baseline on R6000 (RTX PRO 6000 Blackwell)
for the Mac/llama.cpp release track. Results are aligned with Lynn 9B NVFP4 watcher
fields for direct cross-platform comparison.

**Release tracks:**
| Track | Target HW | Model | Engine |
|-------|-----------|-------|--------|
| Lynn NVFP4 | NVIDIA R6000 | Qwen3.5/3.6-9B | Lynn Engine (CUDA) |
| Mac Q4_K_M | Apple Silicon | Qwen3.5-9B | llama.cpp |
| R6000 Q4_K_M | NVIDIA R6000 | Qwen3.5-9B | llama.cpp (this baseline) |

## What It Measures

1. **Single-stream decode TPS** at 128/256/512 max_tokens
2. **Concurrent total TPS** at 2/4/8 parallel requests
3. **Long-context prefill+decode** smoke at 4k/16k/32k characters

## Files

| File | Purpose |
|------|---------|
| `scripts/r6000_qwen35_9b_q4km_llamacpp_baseline.sh` | Main harness (GGUF discovery, server start, bench) |
| `scripts/summarize_qwen35_9b_q4km_baseline.py` | JSON → Markdown summary generator |
| `reports/qwen35_9b/r6000_qwen35_9b_q4km_baseline_pending_sample.json` | PENDING report template |

## Quick Start (R6000)

```bash
# 1. Ensure GGUF on disk
ls -lh /root/autodl-tmp/models/Qwen3.5-9B-Q4_K_M.gguf

# 2. Run baseline
bash scripts/r6000_qwen35_9b_q4km_llamacpp_baseline.sh

# 3. Results in reports/qwen35_9b/
ls -la reports/qwen35_9b/r6000_qwen35_9b_q4km_baseline_*.json
```

## GGUF Download (if missing)

```bash
# HuggingFace CLI (recommended)
huggingface-cli download Qwen/Qwen3.5-9B-GGUF qwen3.5-9b-q4_k_m.gguf \
  --local-dir /root/autodl-tmp/models/

# ModelScope (China mirror)
modelscope download Qwen/Qwen3.5-9B-GGUF qwen3.5-9b-q4_k_m.gguf \
  --local_dir /root/autodl-tmp/models/

# Direct URL
export GGUF_URL="https://huggingface.co/Qwen/Qwen3.5-9B-GGUF/resolve/main/qwen3.5-9b-q4_k_m.gguf"
bash scripts/r6000_qwen35_9b_q4km_llamacpp_baseline.sh
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ROOT` | `/root/autodl-tmp/models` | Model search root |
| `GGUF` | (auto-discovered) | Explicit GGUF path |
| `HF_REPO` | (none) | HuggingFace repo ID for download |
| `MS_REPO` | (none) | ModelScope repo ID for download |
| `GGUF_URL` | (none) | Direct download URL |
| `REPORT_ROOT` | `/root/autodl-tmp/reports/qwen35_9b` | Report output dir |
| `PORT` | `18197` | llama-server port |
| `CTX_SIZE` | `32768` | Context window size |
| `PARALLEL` | `8` | Server parallel slots |
| `SINGLE_MAX_TOKENS` | `128 256 512` | Single-stream decode lengths |
| `CONCURRENCY` | `2 4 8` | Concurrent request counts |
| `LONG_CONTEXT_CHARS` | `4096 16384 32768` | Long-context char counts |

## GGUF Auto-Discovery Order

1. `$GGUF` env var (if set and file exists)
2. `$MODEL_ROOT/Qwen3.5-9B-Q4_K_M.gguf` (standard name)
3. Glob: `$MODEL_ROOT/*Qwen3.5*9B*Q4*K*M*.gguf`

If not found → PENDING_DOWNLOAD report with download command templates.

## llama.cpp Binary Auto-Discovery

1. `/root/autodl-tmp/llama.cpp/build-cuda/bin/llama-server`
2. `/root/autodl-tmp/llama.cpp/build/bin/llama-server`
3. `/root/autodl-tmp/llama.cpp/build/tools/server/llama-server`

Override: `LLAMA_SERVER=/path/to/llama-server`

## JSON Report Schema

```json
{
  "schema": "lynn-qwen35-9b-q4km-baseline-v1",
  "status": "DONE | PENDING_DOWNLOAD",
  "model_id": "Qwen3.5-9B",
  "quant": "Q4_K_M",
  "model_path": "/path/to/gguf",
  "size_gib": "5.20",
  "llama_cpp_binary": "/path/to/llama-server",
  "git_rev": "abc1234",
  "single_tps": {
    "128": {"ok": true, "wall_tps": 180.5, "prompt_tokens": 512, "completion_tokens": 128, "elapsed_s": 0.709},
    "256": {"ok": true, "wall_tps": 175.2, ...},
    "512": {"ok": true, "wall_tps": 168.8, ...}
  },
  "concurrent_tps": {
    "2": {"ok": true, "batch_wall_tps": 340.1, "elapsed_s": 1.505},
    "4": {"ok": true, "batch_wall_tps": 520.4, ...},
    "8": {"ok": true, "batch_wall_tps": 680.2, ...}
  },
  "long_context": {
    "4096": {"ok": true, "prompt_tokens": 1024, "wall_tps": 160.0, ...},
    "16384": {"ok": true, "prompt_tokens": 4096, "wall_tps": 140.5, ...},
    "32768": {"ok": false, "error": "OOM", ...}
  },
  "errors": []
}
```

## Cross-Reference: Lynn NVFP4 Watcher Fields

| Q4_K_M Baseline Field | Lynn NVFP4 Watcher Field | Comparison |
|------------------------|--------------------------|------------|
| `single_tps.512.wall_tps` | `single_tps` (P25 probe) | Direct TPS comparison |
| `concurrent_tps.8.batch_wall_tps` | `concurrent_total_tps` | Concurrency scaling |
| `long_context.32768.ok` | `long_context_ok` | Context window support |

## Behavior Matrix

| Scenario | Expected Output |
|----------|----------------|
| GGUF missing, no download vars | PENDING_DOWNLOAD with download commands |
| GGUF found, server starts | Full benchmark suite |
| Long-context OOM | Recorded as error in report, not fatal |
| Server fails to start | Error in report + exit code |
