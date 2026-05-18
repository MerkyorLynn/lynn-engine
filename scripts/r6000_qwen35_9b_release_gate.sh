#!/usr/bin/env bash
set -euo pipefail

# GPU-free release gate for Qwen3.5-9B.
#
# Aggregates existing BF16, Q4_K_M, and Lynn-native NVFP4 benchmark reports and
# checks artifact presence/sizes on the current host.  Designed to run on R6000,
# but also works locally by marking remote-only artifacts as REPORTED_PRESENT or
# PENDING_ARTIFACT_CHECK instead of failing silently.

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/models}"
REPORT_DIR="${REPORT_DIR:-$ROOT/reports/qwen35_9b}"
OUTPUT_JSON="${OUTPUT_JSON:-$REPORT_DIR/qwen35_9b_release_gate_summary.json}"
OUTPUT_MD="${OUTPUT_MD:-$ROOT/docs/QWEN35_9B_RELEASE_STATUS_20260519.md}"
RELEASE_MATRIX_JSON="${RELEASE_MATRIX_JSON:-$REPORT_DIR/qwen35_9b_release_matrix.json}"

cd "$ROOT"
mkdir -p "$REPORT_DIR" "$(dirname "$OUTPUT_MD")"

"$PYTHON_BIN" scripts/summarize_qwen35_9b_release_gate.py \
  --report-dir "$REPORT_DIR" \
  --model-root "$MODEL_ROOT" \
  --release-matrix-json "$RELEASE_MATRIX_JSON" \
  --output-json "$OUTPUT_JSON" \
  --output-md "$OUTPUT_MD"

echo "[qwen35-9b-release-gate] wrote $OUTPUT_JSON"
echo "[qwen35-9b-release-gate] wrote $OUTPUT_MD"
