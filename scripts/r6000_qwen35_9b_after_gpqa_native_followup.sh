#!/usr/bin/env bash
set -euo pipefail

# Wait for the long Qwen3.5-9B GPQA thinking32 eval to finish, then run the
# GPU-follow-up gates that should not compete with the eval.

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine-main}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
STAMP_BASE="${STAMP_BASE:-$(date +%Y%m%d_%H%M%S)_after_gpqa}"
JSONL="${JSONL:-${REPORT_DIR}/thinking32/qwen35_9b_q4km_gpqa_thinking32_20260519_182901.jsonl}"
LOG="${LOG:-${REPORT_DIR}/after_gpqa_native_followup_${STAMP_BASE}.log}"
POLL_SECONDS="${POLL_SECONDS:-300}"

mkdir -p "$REPORT_DIR"

{
  echo "[after-gpqa] start $(date -Iseconds)"
  echo "[after-gpqa] root=$ROOT"
  echo "[after-gpqa] jsonl=$JSONL"
  echo "[after-gpqa] waiting for openai_mcq_thinking32_eval.py to exit"

  while pgrep -f openai_mcq_thinking32_eval.py >/dev/null 2>&1; do
    lines="$(wc -l < "$JSONL" 2>/dev/null || echo 0)"
    echo "[after-gpqa] still running; jsonl lines=$lines; $(date -Iseconds)"
    sleep "$POLL_SECONDS"
  done

  echo "[after-gpqa] GPQA eval no longer running $(date -Iseconds)"
  cd "$ROOT"
  export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

  echo "[after-gpqa] final P201 summary"
  JSONL="$JSONL" STAMP="${STAMP_BASE}_summary" \
    bash scripts/r6000_qwen35_9b_gpqa_thinking32_summarize.sh || true

  echo "[after-gpqa] P198 native FP4 build preflight"
  ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" REPORT_DIR="$REPORT_DIR" STAMP="${STAMP_BASE}_p198" \
    bash scripts/r6000_qwen35_9b_native_fp4_build_preflight.sh || true

  echo "[after-gpqa] P197b drift isolation safe"
  ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" REPORT_DIR="$REPORT_DIR" STAMP="${STAMP_BASE}_p197b" MAX_NEW="${MAX_NEW:-8}" LIMIT="${LIMIT:-5}" \
    bash scripts/r6000_qwen35_9b_p197b_drift_isolation_safe.sh || true

  echo "[after-gpqa] done $(date -Iseconds)"
} >> "$LOG" 2>&1

echo "$LOG"
