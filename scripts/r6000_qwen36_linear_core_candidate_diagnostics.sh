#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/lynn-engine}
PY=${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}
REPORT_DIR=${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
FIXTURES=${FIXTURES:-$REPORT_DIR/p169_linear_core_fixtures_official_w4a16_20260519_0750}
CANDIDATE_OUTPUT_DIR=${CANDIDATE_OUTPUT_DIR:-}
REPORT=${REPORT:-$REPORT_DIR/p172_linear_core_candidate_diagnostics_${STAMP}.json}
REQUIRED_CANDIDATE_KEYS=${REQUIRED_CANDIDATE_KEYS:-linear_core_out,conv_state_out,recurrent_state_out}
REQUIRE_HASH_MATCH=${REQUIRE_HASH_MATCH:-0}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

mkdir -p "$REPORT_DIR"

args=(
  benchmarks/p172_qwen36_linear_core_candidate_diagnostics.py
  --fixtures "$FIXTURES"
  --out "$REPORT"
  --required-candidate-keys "$REQUIRED_CANDIDATE_KEYS"
)

if [[ -n "$CANDIDATE_OUTPUT_DIR" ]]; then
  args+=(--candidate-output-dir "$CANDIDATE_OUTPUT_DIR")
fi

if [[ "$REQUIRE_HASH_MATCH" == "1" ]]; then
  args+=(--require-hash-match)
fi

"$PY" "${args[@]}"

echo "[p172] fixtures: $FIXTURES"
if [[ -n "$CANDIDATE_OUTPUT_DIR" ]]; then
  echo "[p172] candidate-output-dir: $CANDIDATE_OUTPUT_DIR"
fi
echo "[p172] report: $REPORT"
