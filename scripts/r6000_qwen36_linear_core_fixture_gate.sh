#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/lynn-engine}
PY=${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}
MODEL=${MODEL:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}
REPORT_DIR=${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
LAYERS=${LAYERS:-first-of-block}
FIXTURES=${FIXTURES:-$REPORT_DIR/p169_linear_core_fixtures_official_w4a16_${STAMP}}
OUT=${OUT:-$REPORT_DIR/p169_linear_core_fixture_contract_${STAMP}.json}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Match the current safe Qwen3.6 linear-core profile. This is an admission
# fixture gate only; resident serving remains unchanged.
export LYNN_PREFILL_WARMUP=1
export LYNN_MOE_IMPL=packed_nvfp4
export LYNN_MOE_FAST_FIXED=1
export LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode
export LYNN_NATIVE_DOWN_BACKEND=triton
export LYNN_NATIVE_ACTIVE_MOE_BACKEND=triton
export LYNN_NATIVE_FP4_LM_HEAD=1
export LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1
export LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
export LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
export LYNN_LINEAR_ATTN_GQA_RECURRENT=1
export LYNN_LINEAR_ATTN_CONV_BACKEND=triton_torch_silu
export LYNN_LINEAR_STATE_UPDATE=inplace
export LYNN_RMSNORM_GATED_BACKEND=triton
export LYNN_DECODE_FAST_DISPATCH=1

mkdir -p "$REPORT_DIR"

"$PY" benchmarks/p169_qwen36_linear_core_fixture_contract.py \
  --model "$MODEL" \
  --fixtures "$FIXTURES" \
  --out "$OUT" \
  --layers "$LAYERS" \
  --export \
  --check

echo "[p169] fixtures: $FIXTURES"
echo "[p169] report: $OUT"
