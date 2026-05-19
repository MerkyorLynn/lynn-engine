#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/reports}"
REPORT_DIR="${REPORT_DIR:-${REPORT_ROOT}/qwen36_35b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${REPORT_DIR}/p179_qwen36_35b_full_attn_tail_graph_probe_${STAMP}.json}"
LOG="${LOG:-${REPORT_DIR}/p179_qwen36_35b_full_attn_tail_graph_probe_${STAMP}.log}"
PROMPT="${PROMPT:-用一句话解释 MoE 推理里为什么要先保证数值严格再谈速度。}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-200}"
MAX_LAYERS="${MAX_LAYERS:-10}"
LAYER="${LAYER:--1}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
FAIL_ON_INVESTIGATE="${FAIL_ON_INVESTIGATE:-0}"

mkdir -p "${REPORT_DIR}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export LYNN_FULL_ATTN_ROPE_CACHE="${LYNN_FULL_ATTN_ROPE_CACHE:-1}"
export LYNN_FULL_ATTN_ROPE_CACHE_MAX_SEQ="${LYNN_FULL_ATTN_ROPE_CACHE_MAX_SEQ:-65536}"
export LYNN_QK_NORM_ROPE_BACKEND="${LYNN_QK_NORM_ROPE_BACKEND:-triton_pair}"
export LYNN_RMSNORM_GATED_BACKEND="${LYNN_RMSNORM_GATED_BACKEND:-triton}"
export LYNN_MOE_FAST_FIXED="${LYNN_MOE_FAST_FIXED:-1}"

args=(
  benchmarks/p179_qwen36_35b_full_attn_tail_graph_probe.py
  --model "${MODEL}"
  --out "${OUT}"
  --prompt "${PROMPT}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --max-seq-len "${MAX_SEQ_LEN}"
  --warmup "${WARMUP}"
  --iters "${ITERS}"
  --max-layers "${MAX_LAYERS}"
  --layer "${LAYER}"
)

if [[ "${FAIL_ON_INVESTIGATE}" == "1" ]]; then
  args+=(--fail-on-investigate)
fi

{
  echo "[p179] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[p179] repo=${REPO_ROOT}"
  echo "[p179] model=${MODEL}"
  echo "[p179] out=${OUT}"
  echo "[p179] layer=${LAYER} max_layers=${MAX_LAYERS} warmup=${WARMUP} iters=${ITERS}"
  "${PYTHON_BIN}" "${args[@]}"
  echo "[p179] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "${LOG}"

echo "${OUT}"
