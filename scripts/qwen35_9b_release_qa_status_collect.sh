#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STATUS_FILE="${STATUS_FILE:-reports/qwen35_9b/QWEN35_9B_RELEASE_QA_STATUS_20260519.md}"
REPORT_DIR="${REPORT_DIR:-reports/qwen35_9b}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT"
"$PYTHON_BIN" scripts/qwen35_9b_release_qa_status_collect.py \
  --root "$ROOT" \
  --status-file "$STATUS_FILE" \
  --report-dir "$REPORT_DIR"
