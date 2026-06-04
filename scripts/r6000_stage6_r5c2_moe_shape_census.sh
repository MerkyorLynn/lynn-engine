#!/usr/bin/env bash
# R6000 wrapper for Stage 6 R5-C2 MoE-shape source census.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUTLASS_DIR="${CUTLASS_DIR:-/root/autodl-tmp/src/cutlass}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT}/reports/stage6/r5c2_moe_shape_census_${STAMP}}"

mkdir -p "$OUT_DIR"
git -C "$CUTLASS_DIR" rev-parse HEAD > "$OUT_DIR/cutlass_head.txt" 2>&1 || true
git -C "$CUTLASS_DIR" status --short > "$OUT_DIR/cutlass_status.txt" 2>&1 || true

set +e
"$PYTHON_BIN" "$ROOT/scripts/r6000_stage6_r5c2_moe_shape_census.py" \
  --cutlass-dir "$CUTLASS_DIR" \
  --out "$OUT_DIR/result.json" \
  "$@" | tee "$OUT_DIR/run.log"
probe_rc=${PIPESTATUS[0]}
set -e

summary_rc=0
"$PYTHON_BIN" "$ROOT/scripts/summarize_stage6_r5c2_moe_shape_census.py" \
  "$OUT_DIR/result.json" \
  --markdown-out "$OUT_DIR/summary.md" \
  --strict-exit || summary_rc=$?

echo "[r5c2] artifact=$OUT_DIR"
if [[ "$probe_rc" -ne 0 ]]; then
  exit "$probe_rc"
fi
exit "$summary_rc"
