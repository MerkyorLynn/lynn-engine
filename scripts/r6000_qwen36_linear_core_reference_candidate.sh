#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/lynn-engine}
PY=${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}
MODEL=${MODEL:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}
REPORT_DIR=${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
FIXTURES=${FIXTURES:-$REPORT_DIR/p169_linear_core_fixtures_official_w4a16_20260519_0750}
CANDIDATE_OUTPUT_DIR=${CANDIDATE_OUTPUT_DIR:-$REPORT_DIR/p172_linear_core_reference_candidate_${STAMP}}
REPORT=${REPORT:-$REPORT_DIR/p172_linear_core_reference_candidate_${STAMP}.json}
P169_OUT=${P169_OUT:-$REPORT_DIR/p172_linear_core_reference_candidate_${STAMP}_p169_check.json}
DEVICE=${DEVICE:-cuda}
ONLY_FINAL=${ONLY_FINAL:-1}
REQUIRE_ALL_KEYS=${REQUIRE_ALL_KEYS:-0}
SKIP_P169_CHECK=${SKIP_P169_CHECK:-0}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=${LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4:-1}
export LYNN_LINEAR_ATTN_CONV_BACKEND=${LYNN_LINEAR_ATTN_CONV_BACKEND:-triton_torch_silu}
export LYNN_LINEAR_ATTN_RECURRENT_BACKEND=${LYNN_LINEAR_ATTN_RECURRENT_BACKEND:-triton_fused_prepare}
export LYNN_LINEAR_ATTN_RECURRENT_INPLACE=${LYNN_LINEAR_ATTN_RECURRENT_INPLACE:-1}
export LYNN_LINEAR_ATTN_GQA_RECURRENT=${LYNN_LINEAR_ATTN_GQA_RECURRENT:-1}
export LYNN_RMSNORM_GATED_BACKEND=${LYNN_RMSNORM_GATED_BACKEND:-triton}

mkdir -p "$REPORT_DIR"

args=(
  benchmarks/p172_qwen36_linear_core_reference_candidate.py
  --model "$MODEL"
  --fixtures "$FIXTURES"
  --candidate-output-dir "$CANDIDATE_OUTPUT_DIR"
  --report "$REPORT"
  --p169-out "$P169_OUT"
  --device "$DEVICE"
)

if [[ "$ONLY_FINAL" == "1" ]]; then
  args+=(--only-final)
else
  if [[ "$REQUIRE_ALL_KEYS" == "1" ]]; then
    args+=(--require-all-keys)
  fi
fi

if [[ "$SKIP_P169_CHECK" == "1" ]]; then
  args+=(--skip-p169-check)
fi

"$PY" "${args[@]}"

echo "[p172] fixtures: $FIXTURES"
echo "[p172] candidate-output-dir: $CANDIDATE_OUTPUT_DIR"
echo "[p172] helper report: $REPORT"
echo "[p172] p169 check report: $P169_OUT"

