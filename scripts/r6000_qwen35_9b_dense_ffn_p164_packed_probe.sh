#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
FIXTURE_DIR="${FIXTURE_DIR:-}"
CANDIDATE_DIR="${CANDIDATE_DIR:-${REPORT_DIR}/p164_dense_ffn_packed_candidate_outputs_${STAMP}}"
P164_JSON="${P164_JSON:-${REPORT_DIR}/p164_dense_ffn_packed_microprobe_${STAMP}.json}"
BACKEND="${BACKEND:-auto}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
OUTPUT_DTYPE="${OUTPUT_DTYPE:-fixture}"
WARMUP="${WARMUP:-8}"
REPEAT="${REPEAT:-32}"
PREPARE_NATIVE="${PREPARE_NATIVE:-1}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

if [[ -z "$FIXTURE_DIR" ]]; then
  latest_fixture="$(find "$REPORT_DIR" -maxdepth 1 -type d -name 'p159_dense_ffn_fixtures_*' | sort | tail -n 1 || true)"
  if [[ -z "$latest_fixture" ]]; then
    echo "[p164] no P159 fixture dir found under $REPORT_DIR; set FIXTURE_DIR=/path/to/p159_dense_ffn_fixtures_*" >&2
    exit 2
  fi
  FIXTURE_DIR="$latest_fixture"
fi

prepare_args=()
if [[ "$PREPARE_NATIVE" == "1" ]]; then
  prepare_args+=(--prepare-native)
fi

echo "[p164] root=$ROOT"
echo "[p164] python=$PYTHON_BIN"
echo "[p164] model=$MODEL"
echo "[p164] fixtures=$FIXTURE_DIR"
echo "[p164] candidate_dir=$CANDIDATE_DIR"
echo "[p164] backend=$BACKEND dtype=$DTYPE output_dtype=$OUTPUT_DTYPE warmup=$WARMUP repeat=$REPEAT prepare_native=$PREPARE_NATIVE"

"$PYTHON_BIN" benchmarks/p164_qwen35_9b_dense_ffn_packed_microprobe.py \
  --fixtures "$FIXTURE_DIR" \
  --model "$MODEL" \
  --out "$CANDIDATE_DIR" \
  --backend "$BACKEND" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --output-dtype "$OUTPUT_DTYPE" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT" \
  --report "$P164_JSON" \
  "${prepare_args[@]}"

echo "[p164] done candidate_dir=$CANDIDATE_DIR"
echo "[p164] p160_candidate_output_dir=$CANDIDATE_DIR"
echo "[p164] report=$P164_JSON"
