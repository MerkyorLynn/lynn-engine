#!/usr/bin/env bash
# P197 · W4A16 vs W4A8 per-step token drift probe
# Run on R6000 with CUDA build.
# Usage: bash scripts/r6000_qwen35_9b_fp4xfp8_token_drift_probe.sh
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MAX_NEW="${MAX_NEW:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
LIMIT="${LIMIT:-5}"
CANDIDATE_PROFILE="${CANDIDATE_PROFILE:-true_fp4xfp8}"
SIDECAR_DIR="${SIDECAR_DIR:-/root/autodl-tmp/reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar}"
OUT="${REPORT_DIR}/p197_fp4xfp8_token_drift_${STAMP}.json"

cd "$ROOT"
mkdir -p "$REPORT_DIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "[p197] root=$ROOT"
echo "[p197] model=$MODEL"
echo "[p197] prompts=$LIMIT  max_new=$MAX_NEW  max_seq_len=$MAX_SEQ_LEN"
echo "[p197] candidate_profile=$CANDIDATE_PROFILE"
echo "[p197] out=$OUT"

"$PYTHON_BIN" benchmarks/p197_qwen35_9b_fp4xfp8_token_drift_probe.py \
  --model "$MODEL" \
  --max-new "$MAX_NEW" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --limit "$LIMIT" \
  --candidate-profile "$CANDIDATE_PROFILE" \
  --sidecar-dir "$SIDECAR_DIR" \
  --out "$OUT"

echo "[p197] done report=$OUT"
