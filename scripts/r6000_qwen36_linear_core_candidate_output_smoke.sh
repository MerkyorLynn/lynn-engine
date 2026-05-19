#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/lynn-engine}
PY=${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}
REPORT_DIR=${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
FIXTURES=${FIXTURES:-$REPORT_DIR/p169_linear_core_fixtures_official_w4a16_20260519_0750}
CANDIDATE_OUTPUT_DIR=${CANDIDATE_OUTPUT_DIR:-$REPORT_DIR/p171_linear_core_identity_candidate_${STAMP}}
REPORT=${REPORT:-$REPORT_DIR/p171_linear_core_candidate_output_smoke_${STAMP}.json}
P169_OUT=${P169_OUT:-$REPORT_DIR/p171_linear_core_candidate_output_smoke_${STAMP}_p169_check.json}
DEVICE=${DEVICE:-cpu}
ONLY_FINAL=${ONLY_FINAL:-0}
REQUIRE_ALL_KEYS=${REQUIRE_ALL_KEYS:-1}
SKIP_P169_CHECK=${SKIP_P169_CHECK:-0}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

mkdir -p "$REPORT_DIR"

args=(
  benchmarks/p171_qwen36_linear_core_candidate_output_smoke.py
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

echo "[p171] fixtures: $FIXTURES"
echo "[p171] candidate-output-dir: $CANDIDATE_OUTPUT_DIR"
echo "[p171] helper report: $REPORT"
echo "[p171] p169 check report: $P169_OUT"
