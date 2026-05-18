#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_JSON="${OUT_JSON:-${REPORT_DIR}/p155_qwen35_9b_dense_ffn_phase_profile_${STAMP}.json}"
MAX_NEW="${MAX_NEW:-128}"
SKIP_STEPS="${SKIP_STEPS:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
TOP_K="${TOP_K:-0}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "[p155] root=$ROOT"
echo "[p155] python=$PYTHON_BIN"
echo "[p155] model=$MODEL"
echo "[p155] max_new=$MAX_NEW skip_steps=$SKIP_STEPS max_seq_len=$MAX_SEQ_LEN"
echo "[p155] out=$OUT_JSON"
echo "[p155] runtime env is caller-controlled; this wrapper does not set Lynn fast/default knobs"

"$PYTHON_BIN" benchmarks/p155_qwen35_9b_dense_ffn_phase_profile.py \
  --model "$MODEL" \
  --max-new "$MAX_NEW" \
  --skip-steps "$SKIP_STEPS" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --top-k "$TOP_K" \
  --out "$OUT_JSON"

echo "[p155] done report=$OUT_JSON"
