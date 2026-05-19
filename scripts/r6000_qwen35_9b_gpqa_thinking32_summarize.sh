#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# R6000: P201 GPQA Diamond thinking32 live summarizer
#
# Summarizes the in-progress Q4_K_M GPQA eval JSONL.
# Safe to run repeatedly while eval is running.
#
# Usage:
#   bash scripts/r6000_qwen35_9b_gpqa_thinking32_summarize.sh
#   bash scripts/r6000_qwen35_9b_gpqa_thinking32_summarize.sh --watch 30
# ─────────────────────────────────────────────────────────────

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine-main}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

# Auto-discover the most recent thinking32 GPQA JSONL
JSONL="${JSONL:-}"
if [[ -z "$JSONL" ]]; then
  JSONL="$(ls -t "$REPORT_DIR"/thinking32/*thinking32*gpqa*.jsonl "$REPORT_DIR"/thinking32/*gpqa*thinking32*.jsonl 2>/dev/null | head -1 || true)"
fi
if [[ -z "$JSONL" ]]; then
  JSONL="$(ls -t "$REPORT_DIR"/*thinking32*gpqa*.jsonl "$REPORT_DIR"/*q4km*gpqa*.jsonl 2>/dev/null | head -1 || true)"
fi
if [[ -z "$JSONL" ]]; then
  JSONL="$(ls -t "$REPORT_DIR"/*gpqa*.jsonl 2>/dev/null | head -1 || true)"
fi

if [[ -z "$JSONL" || ! -f "$JSONL" ]]; then
  echo "[p201] ERROR: No GPQA JSONL found in $REPORT_DIR" >&2
  echo "[p201] Set JSONL=/path/to/file.jsonl" >&2
  exit 1
fi

OUT_JSON="${REPORT_DIR}/p201_gpqa_live_summary_${STAMP}.json"

cd "$ROOT"
echo "[p201] JSONL: $JSONL ($(wc -l < "$JSONL") lines)"
echo "[p201] Output: $OUT_JSON"
echo ""

exec "$PYTHON_BIN" scripts/p201_gpqa_thinking32_live_summarizer.py \
  "$JSONL" \
  --out "$OUT_JSON" \
  "$@"
