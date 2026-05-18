#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/reports}"
REPORT_DIR="${REPORT_DIR:-${REPORT_ROOT}/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${REPORT_DIR}/p174_qwen35_9b_full_attn_tail_graph_probe_${STAMP}.json}"
LOG="${LOG:-${REPORT_DIR}/p174_qwen35_9b_full_attn_tail_graph_probe_${STAMP}.log}"
PROMPT="${PROMPT:-用一句话解释 CUDA graph 为什么适合固定形状的推理尾部。}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-200}"
MAX_LAYERS="${MAX_LAYERS:-1}"
LAYER="${LAYER:--1}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
FAIL_ON_INVESTIGATE="${FAIL_ON_INVESTIGATE:-0}"

mkdir -p "${REPORT_DIR}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

args=(
  benchmarks/p174_qwen35_9b_full_attn_tail_graph_probe.py
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
  echo "[p174] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[p174] repo=${REPO_ROOT}"
  echo "[p174] model=${MODEL}"
  echo "[p174] out=${OUT}"
  echo "[p174] layer=${LAYER} max_layers=${MAX_LAYERS} warmup=${WARMUP} iters=${ITERS}"
  "${PYTHON_BIN}" "${args[@]}"
  echo "[p174] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "${LOG}"

echo "${OUT}"
