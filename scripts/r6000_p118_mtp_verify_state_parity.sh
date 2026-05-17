#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/lynn-engine}
PY=${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}
MODEL=${MODEL:-/root/autodl-tmp/models/lynn-27b-a3b-w4a8-nvfp4-v2}
REPORT_ROOT=${REPORT_ROOT:-/root/autodl-tmp/reports}
PROMPTS_FILE=${PROMPTS_FILE:-}
MAX_EVENTS=${MAX_EVENTS:-8}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
DTYPE=${DTYPE:-bfloat16}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
REPORT_DIR=$REPORT_ROOT/p118
OUT=$REPORT_DIR/p118_mtp_verify_state_parity_r6000_${TS}.json
LOG=$REPORT_DIR/p118_mtp_verify_state_parity_r6000_${TS}.log

mkdir -p "$REPORT_DIR"

cd "$REPO"
export PYTHONPATH="$REPO"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export LYNN_MOE_IMPL=${LYNN_MOE_IMPL:-packed_nvfp4}
export LYNN_NATIVE_FP4_LM_HEAD=${LYNN_NATIVE_FP4_LM_HEAD:-1}
export LYNN_MTP_SHADOW_VERIFY=0
export LYNN_MTP_VERIFY=0
export LYNN_FULL_TOKEN_GRAPH_SLOT=0

{
  echo "[r6000-p118] start $(date)"
  echo "[r6000-p118] model=$MODEL"
  echo "[r6000-p118] max_events=$MAX_EVENTS max_seq_len=$MAX_SEQ_LEN dtype=$DTYPE"
  echo "[r6000-p118] native_fp4_lm_head=$LYNN_NATIVE_FP4_LM_HEAD moe_impl=$LYNN_MOE_IMPL"

  args=(
    benchmarks/p118_mtp_verify_state_parity.py
    --base-model "$MODEL"
    --out "$OUT"
    --device cuda
    --dtype "$DTYPE"
    --max-seq-len "$MAX_SEQ_LEN"
    --max-events "$MAX_EVENTS"
  )
  if [[ -n "$PROMPTS_FILE" ]]; then
    args+=(--prompts-file "$PROMPTS_FILE")
  fi

  "$PY" "${args[@]}"
  echo "[r6000-p118] report=$OUT"
  echo "[r6000-p118] done $(date)"
} >> "$LOG" 2>&1

echo "$LOG"
