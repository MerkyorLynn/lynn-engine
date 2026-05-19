#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PY="${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
PACKED_FIXTURES="${PACKED_FIXTURES:-/root/autodl-tmp/lynn-engine/reports/qwen36_35b/p133_fixtures}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$ROOT/reports/qwen36_35b/p177_qwen36_router_linear_out_probe_${STAMP}.json}"

cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export LYNN_MOE_IMPL=packed_nvfp4
export LYNN_MOE_FAST_FIXED=1
export LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode
export LYNN_NATIVE_DOWN_BACKEND=triton
export LYNN_NATIVE_ACTIVE_MOE_BACKEND=triton
export LYNN_PACKED_SHARED_EXPERT=0
export LYNN_ROUTER_TOPK_SORTED=0

"$PY" benchmarks/p177_qwen36_router_linear_out_probe.py \
  --model "$MODEL" \
  --packed-fixtures "$PACKED_FIXTURES" \
  --warmup "${WARMUP:-30}" \
  --iters "${ITERS:-160}" \
  --max-fixtures "${MAX_FIXTURES:-18}" \
  --out "$OUT"

echo "[p177] wrote $OUT"
