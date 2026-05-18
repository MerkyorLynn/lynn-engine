#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
FIXTURE_DIR="${FIXTURE_DIR:-}"
CANDIDATE_DIR="${CANDIDATE_DIR:-${REPORT_DIR}/p161_dense_ffn_candidate_outputs_${STAMP}}"
P161_JSON="${P161_JSON:-${REPORT_DIR}/p161_dense_ffn_candidate_outputs_${STAMP}.json}"
P160_JSON="${P160_JSON:-${REPORT_DIR}/p161_dense_ffn_candidate_p160_contract_${STAMP}.json}"
BACKEND="${BACKEND:-reference}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
OUTPUT_DTYPE="${OUTPUT_DTYPE:-fixture}"
WARMUP="${WARMUP:-8}"
REPEAT="${REPEAT:-32}"
COMPILE_MODE="${COMPILE_MODE:-default}"
COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

if [[ -z "$FIXTURE_DIR" ]]; then
  latest_fixture="$(find "$REPORT_DIR" -maxdepth 1 -type d -name 'p159_dense_ffn_fixtures_*' | sort | tail -n 1 || true)"
  if [[ -z "$latest_fixture" ]]; then
    echo "[p161] no P159 fixture dir found under $REPORT_DIR; set FIXTURE_DIR=/path/to/p159_dense_ffn_fixtures_*" >&2
    exit 2
  fi
  FIXTURE_DIR="$latest_fixture"
fi

compile_args=()
if [[ "$COMPILE_FULLGRAPH" == "1" ]]; then
  compile_args+=(--compile-fullgraph)
fi

echo "[p161] root=$ROOT"
echo "[p161] python=$PYTHON_BIN"
echo "[p161] model=$MODEL"
echo "[p161] fixtures=$FIXTURE_DIR"
echo "[p161] candidate_dir=$CANDIDATE_DIR"
echo "[p161] backend=$BACKEND dtype=$DTYPE output_dtype=$OUTPUT_DTYPE warmup=$WARMUP repeat=$REPEAT"

"$PYTHON_BIN" benchmarks/p161_qwen35_9b_dense_ffn_candidate_outputs.py \
  --fixtures "$FIXTURE_DIR" \
  --model "$MODEL" \
  --out "$CANDIDATE_DIR" \
  --backend "$BACKEND" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --output-dtype "$OUTPUT_DTYPE" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT" \
  --compile-mode "$COMPILE_MODE" \
  --report "$P161_JSON" \
  "${compile_args[@]}"

"$PYTHON_BIN" benchmarks/p160_qwen35_9b_dense_ffn_fixture_contract.py \
  --fixtures "$FIXTURE_DIR" \
  --model "$MODEL" \
  --candidate-output-dir "$CANDIDATE_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT" \
  --out "$P160_JSON"

echo "[p161] done candidate_dir=$CANDIDATE_DIR"
echo "[p161] p161_report=$P161_JSON"
echo "[p161] p160_contract=$P160_JSON"
