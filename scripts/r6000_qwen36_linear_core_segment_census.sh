#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/lynn-engine}
PY=${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}
MODEL=${MODEL:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}
REPORT_DIR=${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
LAYERS=${LAYERS:-all}
WARMUP=${WARMUP:-8}
ITERS=${ITERS:-120}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Match the current safe Qwen3.6 W4A16 profile. This script only measures
# segment latency; it does not promote or alter defaults.
export LYNN_PREFILL_WARMUP=1
export LYNN_MOE_IMPL=packed_nvfp4
export LYNN_MOE_FAST_FIXED=1
export LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode
export LYNN_NATIVE_DOWN_BACKEND=triton
export LYNN_NATIVE_ACTIVE_MOE_BACKEND=triton
export LYNN_PACKED_DECODE=0
export LYNN_PACKED_DECODE_PREPARE_NATIVE=0
export LYNN_PACKED_SHARED_EXPERT=0
export LYNN_NATIVE_FP4_LM_HEAD=1
export LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1
export LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
export LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
export LYNN_LINEAR_ATTN_GQA_RECURRENT=1
export LYNN_LINEAR_ATTN_CONV_BACKEND=triton_torch_silu
export LYNN_LINEAR_BLOCK_GRAPH=1
export LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
export LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
export LYNN_LINEAR_STATE_UPDATE=inplace
export LYNN_QK_NORM_ROPE_BACKEND=triton_pair
export LYNN_FULL_ATTN_ROPE_CACHE=1
export LYNN_FULL_ATTN_ROPE_CACHE_MAX_SEQ=65536
export LYNN_RMSNORM_GATED_BACKEND=triton
export LYNN_DECODE_FAST_DISPATCH=1
export LYNN_SHARED_EXPERT_GATE_BACKEND=torch
export LYNN_ROUTER_TOPK_OUT_BUFFER=1

mkdir -p "$REPORT_DIR"
OUT="$REPORT_DIR/p168_qwen36_linear_core_segment_census_${STAMP}.json"

"$PY" benchmarks/p168_qwen36_linear_core_segment_census.py \
  --model "$MODEL" \
  --out "$OUT" \
  --layers "$LAYERS" \
  --warmup "$WARMUP" \
  --iters "$ITERS"

echo "[p168] wrote $OUT"
