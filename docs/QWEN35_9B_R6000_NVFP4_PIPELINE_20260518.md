# Qwen3.5-9B R6000 NVFP4 Pipeline

**Branch:** `kimi/qwen35-9b-r6000-official-pack-bench-20260518`  
**Date:** 2026-05-18  
**Status:** Scaffold (DRY_RUN=1 default)

---

## Overview

This pipeline provides the full R6000 benchmark harness for **Qwen3.5-9B** official assets across three quantization paths:

| Asset | Path | Expected Size | Status |
|-------|------|---------------|--------|
| BF16 | `/root/autodl-tmp/models/Qwen3.5-9B-BF16` | ~18 GiB | Waiting for index |
| NVFP4 | `/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0` | ~5.5 GiB | Waiting for manifest |
| Q4_K_M | `/root/autodl-tmp/models/Qwen3.5-9B-Q4_K_M.gguf` | ~5.2 GiB | PENDING (no download) |

**Important:** This is the official **Qwen3.5-9B** route. The existing `r6000_qwen36_9b_official_w4a16_pack.sh` script name is a legacy artifact — actual R6000 env points to `Qwen3.5-9B` paths.

**Product positioning:**
- **Qwen3.5-9B** = endpoint / 16G VRAM branch candidate (official release).
- **BF16** = quality ceiling; ~18 GiB. Reference for all quantized variants.
- **NVFP4 (W4A16)** = NVIDIA Blackwell serving path; ~5.5 GiB. Lynn-native packed decode + TensorCore MMA.
- **Q4_K_M** = Mac / Apple Silicon / llama.cpp path; ~5.2 GiB. No automatic download.

---

## Files

| File | Purpose |
|------|---------|
| `scripts/r6000_qwen35_9b_official_matrix_watch.sh` | Full benchmark harness. Assets → GPU check → MMLU/GPQA/TPS → summary. |
| `scripts/summarize_qwen35_9b_r6000_reports.py` | Renders Markdown from unified summary JSON. Empty-input safe. |
| `reports/qwen35_9b/r6000_9b_official_pipeline_manifest_template.json` | Stage manifest template with matrix field schema. |
| `docs/QWEN35_9B_R6000_NVFP4_PIPELINE_20260518.md` | This document. |

---

## Usage

### 1. Dry-run (default)

```bash
bash scripts/r6000_qwen35_9b_official_matrix_watch.sh
```

Prints all commands. No server starts, no eval runs, no GPU used.

### 2. Execute on R6000 (full pipeline)

```bash
export DRY_RUN=0
export SKIP_GPU=0
export MMLU_RUNNER=/path/to/mmlu_runner_v2.py
export GPQA_RUNNER=/path/to/gpqa_runner_v2.py
export MMLU_DATASET=/tmp/datasets/mmlu
export GPQA_DATASET=/tmp/datasets/gpqa/gpqa_diamond.csv

bash scripts/r6000_qwen35_9b_official_matrix_watch.sh
```

### 3. Summarize after run

```bash
python3 scripts/summarize_qwen35_9b_r6000_reports.py \
    --summary /root/autodl-tmp/reports/qwen35_9b/r6000_qwen35_9b_official_matrix_summary_*.json \
    --out docs/QWEN35_9B_R6000_NVFP4_PIPELINE_20260518.md
```

---

## Pipeline Stages

### Stage 0: Asset completion check

Verifies:
- BF16 `model.safetensors.index.json`
- NVFP4 `lynn_quant_manifest.json`
- Q4_K_M GGUF (optional)

### Stage 1: Size / manifest summary

- BF16 total bytes → GiB
- NVFP4 total bytes → GiB
- Manifest fields: `quantized_count`, `kept_count`, `output_shards`, `pack_elapsed_seconds`

### Stage 2: GPU idle check

- Queries `nvidia-smi` memory usage
- If any GPU > 1000 MB used, skips all GPU benchmarks
- Respects `SKIP_GPU=1`

**Why:** Never competes with 35B P37 GPU work.

### Stage 3: Per-quant benchmark

For each ready quant (BF16, NVFP4):

1. **Start server** — `python -m server.openai_http --model $MODEL --port $PORT`
2. **MMLU-500-5shot** — via `MMLU_RUNNER` (e.g. `mmlu_runner_v2.py`)
3. **GPQA-diamond** — via `GPQA_RUNNER` (e.g. `gpqa_runner_v2.py`)
4. **Single TPS** — `benchmarks/p25_server_decode_tps_probe.py` at 128/256/512 tokens
5. **Stop server**

**Blocked reasons:**
- Model asset not ready → PENDING
- GPU not idle → BLOCKED
- MMLU/GPQA runner not found → BLOCKED
- P25 probe not found → BLOCKED

For Q4_K_M:
- If GGUF missing → PENDING (no download)
- If present → currently PENDING (llama.cpp eval not in this harness)

### Stage 4: Unified summary JSON

Writes `r6000_qwen35_9b_official_matrix_summary_${STAMP}.json` with schema:

```json
{
  "results": {
    "bf16": {
      "status": "DONE",
      "size_gib": 18.0,
      "load_seconds": 45.2,
      "mmlu_500_5shot": {
        "score": 0.825, "correct": 412, "total": 500,
        "report_path": "...", "status": "DONE", "blocked_reason": null
      },
      "gpqa_diamond": {
        "score": 0.452, "correct": 23, "total": 50,
        "report_path": "...", "status": "DONE", "blocked_reason": null
      },
      "single_tps": {
        "tps_128": 120.5, "tps_256": 115.3, "tps_512": 108.7,
        "report_path": "...", "status": "DONE", "blocked_reason": null
      }
    }
  }
}
```

### Stage 5: Markdown report

Renders quality + performance tables with status badges.

---

## Safety Guarantees

- **DRY_RUN=1 by default.** No commands execute unless explicitly set to `0`.
- **No downloads.** BF16/NVFP4 must be pre-present; Q4_K_M is PENDING if missing.
- **GPU-aware.** Checks `nvidia-smi` before any model loading. Respects `SKIP_GPU=1`.
- **Server cleanup.** `trap` ensures server is killed on script exit.
- **No deletions.** Only creates report files; never removes assets.
- **Empty-input safe.** Summarizer produces valid scaffold even with zero report files.
- **Does not touch 35B files.** No modification to `csrc/*`, `engine/moe*`, `engine/resident_runner.py`, `server/*`, or promotion gates.

---

## Backfill Checklist

- [ ] BF16 download complete → `model.safetensors.index.json` present
- [ ] NVFP4 pack complete → `lynn_quant_manifest.json` present
- [ ] Install MMLU/GPQA runners on R6000 (if not already present)
- [ ] Run watcher with `DRY_RUN=0 SKIP_GPU=0`
- [ ] All three metrics GREEN for BF16 and NVFP4
- [ ] Summarize → Markdown report
- [ ] (Optional) Q4_K_M GGUF available → mark READY

---

*Scaffold only — no model inference in DRY_RUN=1 mode.*
