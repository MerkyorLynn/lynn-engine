#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
PACKED_FIXTURES="${PACKED_FIXTURES:-/root/autodl-tmp/reports/qwen36_35b/p138_packed_slot_fixtures_kimi_20260518}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_JSON="${OUT_JSON:-${REPORT_DIR}/p163_qwen36_router_boundary_probe_${STAMP}.json}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

echo "[p163-wrapper] root=$ROOT"
echo "[p163-wrapper] model=$MODEL"
echo "[p163-wrapper] packed_fixtures=$PACKED_FIXTURES"
echo "[p163-wrapper] out_json=$OUT_JSON"

"$PYTHON_BIN" benchmarks/p163_qwen36_router_boundary_probe.py \
  --model "$MODEL" \
  --packed-fixtures "$PACKED_FIXTURES" \
  --warmup "${WARMUP:-20}" \
  --iters "${ITERS:-120}" \
  --out "$OUT_JSON"

echo "[p163-wrapper] done report=$OUT_JSON"
