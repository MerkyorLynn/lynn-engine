#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
PROMPTS_JSON="${PROMPTS_JSON:-${ROOT}/scripts/qwen36_structured_hard_prompts_70.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LIMIT="${LIMIT:-70}"
MAX_NEW="${MAX_NEW:-64}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
P190_JSON="${P190_JSON:-${REPORT_DIR}/p190_qwen35_9b_true_fp8_resident_gate_${STAMP}.json}"
P197_REPORT="${P197_REPORT:-${REPORT_DIR}/p197_token_drift_probe.json}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "[p190] root=$ROOT"
echo "[p190] model=$MODEL"
echo "[p190] prompts=$PROMPTS_JSON limit=$LIMIT"
echo "[p190] out=$P190_JSON"

P197_ARGS=""
if [[ -f "$P197_REPORT" ]]; then
  P197_ARGS="--p197-report $P197_REPORT"
  echo "[p190] Using P197 drift report: $P197_REPORT"
else
  echo "[p190] P197 report not found at $P197_REPORT — running without drift signal"
fi

"$PYTHON_BIN" benchmarks/p190_qwen35_9b_true_fp8_resident_gate.py \
  --model "$MODEL" \
  --prompts-json "$PROMPTS_JSON" \
  --limit "$LIMIT" \
  --max-new "$MAX_NEW" \
  --max-seq-len "$MAX_SEQ_LEN" \
  $P197_ARGS \
  --out "$P190_JSON"

echo "[p190] done report=$P190_JSON"
