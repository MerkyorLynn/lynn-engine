#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-9B release gate summary — GPU-free aggregator.
#
# Reads existing MMLU/GPQA/TPS/structured report files and emits a unified
# JSON with a three-tier decision: PROMOTE_READY / NEEDS_MORE_DATA / CLOSED.
#
# Usage:
#   bash scripts/qwen35_9b_release_gate_summary.sh
#   bash scripts/qwen35_9b_release_gate_summary.sh --reports-dir /path/to/reports --out /path/to/output.json

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REPORTS_DIR="${REPORTS_DIR:-$ROOT/reports/qwen35_9b}"
OUT="${OUT:-$REPORTS_DIR/qwen35_9b_release_gate_summary_latest.json}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reports-dir)
      REPORTS_DIR="$2"
      shift 2
      ;;
    --out)
      OUT="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

cd "$ROOT"
mkdir -p "$REPORTS_DIR" "$(dirname "$OUT")"

"$PYTHON_BIN" scripts/qwen35_9b_release_gate_summary.py \
  --reports-dir "$REPORTS_DIR" \
  --out "$OUT"

echo "[qwen35-9b-release-gate-summary] done"
