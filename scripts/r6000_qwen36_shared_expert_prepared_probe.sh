#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
PACKED_FIXTURES="${PACKED_FIXTURES:-/root/autodl-tmp/reports/qwen36_35b/p138_packed_slot_fixtures_kimi_20260518}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_JSON="${OUT_JSON:-${REPORT_DIR}/p167_qwen36_shared_expert_prepared_probe_${STAMP}.json}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "[p167] root=$ROOT"
echo "[p167] model=$MODEL"
echo "[p167] fixtures=$PACKED_FIXTURES"
echo "[p167] out=$OUT_JSON"

"$PYTHON_BIN" -m py_compile benchmarks/p167_qwen36_shared_expert_prepared_probe.py
"$PYTHON_BIN" benchmarks/p167_qwen36_shared_expert_prepared_probe.py \
  --model "$MODEL" \
  --packed-fixtures "$PACKED_FIXTURES" \
  --warmup "${WARMUP:-8}" \
  --iters "${ITERS:-40}" \
  --out "$OUT_JSON"

echo "[p167] report=$OUT_JSON"
