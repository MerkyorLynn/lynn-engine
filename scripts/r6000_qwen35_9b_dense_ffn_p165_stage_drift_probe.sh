#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
FIXTURE_DIR="${FIXTURE_DIR:-}"
P165_JSON="${P165_JSON:-${REPORT_DIR}/p165_dense_ffn_stage_drift_${STAMP}.json}"
BACKENDS="${BACKENDS:-scalar_bridge,native_fast_2d}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
WARMUP="${WARMUP:-8}"
REPEAT="${REPEAT:-32}"
PREPARE_NATIVE="${PREPARE_NATIVE:-1}"
LOAD_BACKEND="${LOAD_BACKEND:-native_scaled_mm}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

if [[ -z "$FIXTURE_DIR" ]]; then
  latest_fixture="$(find "$REPORT_DIR" -maxdepth 1 -type d -name 'p159_dense_ffn_fixtures_*' | sort | tail -n 1 || true)"
  if [[ -z "$latest_fixture" ]]; then
    echo "[p165] no P159 fixture dir found under $REPORT_DIR; set FIXTURE_DIR=/path/to/p159_dense_ffn_fixtures_*" >&2
    exit 2
  fi
  FIXTURE_DIR="$latest_fixture"
fi

prepare_args=()
if [[ "$PREPARE_NATIVE" == "1" ]]; then
  prepare_args+=(--prepare-native)
fi

echo "[p165] root=$ROOT"
echo "[p165] python=$PYTHON_BIN"
echo "[p165] model=$MODEL"
echo "[p165] fixtures=$FIXTURE_DIR"
echo "[p165] report=$P165_JSON"
echo "[p165] backends=$BACKENDS dtype=$DTYPE warmup=$WARMUP repeat=$REPEAT prepare_native=$PREPARE_NATIVE load_backend=$LOAD_BACKEND"

"$PYTHON_BIN" benchmarks/p165_qwen35_9b_dense_ffn_stage_drift_probe.py \
  --fixtures "$FIXTURE_DIR" \
  --model "$MODEL" \
  --out "$P165_JSON" \
  --backends "$BACKENDS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT" \
  --load-backend "$LOAD_BACKEND" \
  "${prepare_args[@]}"

echo "[p165] done report=$P165_JSON"
