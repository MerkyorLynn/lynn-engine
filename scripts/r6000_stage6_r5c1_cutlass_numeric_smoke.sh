#!/usr/bin/env bash
# R6000 wrapper for Stage 6 R5-C1 CUTLASS native NVF4 + UE4M3 numeric smoke.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUTLASS_DIR="${CUTLASS_DIR:-/root/autodl-tmp/src/cutlass}"
BUILD_DIR="${BUILD_DIR:-/root/autodl-tmp/src/cutlass/build-r5c1-sm120a-tools-on}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT}/reports/stage6/r5c1_cutlass_numeric_smoke_${STAMP}}"

mkdir -p "$OUT_DIR"
nvidia-smi > "$OUT_DIR/nvidia_smi_before.txt" 2>&1 || true
if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$ROOT" rev-parse HEAD > "$OUT_DIR/lynn_engine_head.txt" 2>&1 || true
  git -C "$ROOT" status --short > "$OUT_DIR/lynn_engine_status.txt" 2>&1 || true
fi
git -C "$CUTLASS_DIR" rev-parse HEAD > "$OUT_DIR/cutlass_head.txt" 2>&1 || true
git -C "$CUTLASS_DIR" status --short > "$OUT_DIR/cutlass_status.txt" 2>&1 || true

set +e
"$PYTHON_BIN" "$ROOT/scripts/r6000_stage6_r5c1_cutlass_numeric_smoke.py" \
  --cutlass-dir "$CUTLASS_DIR" \
  --build-dir "$BUILD_DIR" \
  --build \
  --out "$OUT_DIR/result.json" \
  "$@" | tee "$OUT_DIR/run.log"
probe_rc=${PIPESTATUS[0]}
set -e

summary_rc=0
"$PYTHON_BIN" "$ROOT/scripts/summarize_stage6_r5c1_cutlass_numeric_smoke.py" \
  "$OUT_DIR/result.json" \
  --markdown-out "$OUT_DIR/summary.md" \
  --strict-exit || summary_rc=$?

echo "[r5c1] artifact=$OUT_DIR"
if [[ "$probe_rc" -ne 0 ]]; then
  exit "$probe_rc"
fi
exit "$summary_rc"
