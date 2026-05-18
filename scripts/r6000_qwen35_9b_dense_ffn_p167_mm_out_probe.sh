#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
FIXTURE_DIR="${FIXTURE_DIR:-}"
OUT_JSON="${OUT_JSON:-${REPORT_DIR}/p167_dense_ffn_mm_out_probe_${STAMP}.json}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
WARMUP="${WARMUP:-8}"
REPEAT="${REPEAT:-64}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

if [[ -z "$FIXTURE_DIR" ]]; then
  latest_fixture="$(find "$REPORT_DIR" -maxdepth 1 -type d -name 'p159_dense_ffn_fixtures_*' | sort | tail -n 1 || true)"
  if [[ -z "$latest_fixture" ]]; then
    echo "[p167] no P159 fixture dir found under $REPORT_DIR; set FIXTURE_DIR=/path/to/p159_dense_ffn_fixtures_*" >&2
    exit 2
  fi
  FIXTURE_DIR="$latest_fixture"
fi

echo "[p167] root=$ROOT"
echo "[p167] python=$PYTHON_BIN"
echo "[p167] model=$MODEL"
echo "[p167] fixtures=$FIXTURE_DIR"
echo "[p167] out=$OUT_JSON"

"$PYTHON_BIN" benchmarks/p167_qwen35_9b_dense_ffn_mm_out_probe.py \
  --fixtures "$FIXTURE_DIR" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --warmup "$WARMUP" \
  --repeat "$REPEAT" \
  --out "$OUT_JSON"

echo "[p167] done report=$OUT_JSON"
