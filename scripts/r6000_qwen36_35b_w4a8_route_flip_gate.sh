#!/usr/bin/env bash
# r6000_qwen36_35b_w4a8_route_flip_gate.sh
#
# R6000 runner for P189: Qwen3.6-35B MoE W4A8 route-flip gate.
#
# Locates latest p133 fixture directory, runs p189, writes JSON report.
#
# Usage:
#   bash scripts/r6000_qwen36_35b_w4a8_route_flip_gate.sh
#   FIXTURE_DIR=/path/to/fixtures bash scripts/r6000_qwen36_35b_w4a8_route_flip_gate.sh
#
# Branch: qwen/qwen36-35b-w4a8-route-flip-gate-20260519

set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

FP8_FORMAT="${FP8_FORMAT:-e4m3}"
GRANULARITY="${GRANULARITY:-per16}"
DEVICE="${DEVICE:-cuda}"

FIXTURE_DIR="${FIXTURE_DIR:-}"
P189_JSON="${P189_JSON:-${REPORT_DIR}/p189_moe_w4a8_route_flip_gate_${STAMP}.json}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# ── Locate p133 fixture dir ─────────────────────────────────────────────────
if [[ -z "$FIXTURE_DIR" ]]; then
  FIXTURE_DIR="$(find "$REPORT_DIR" -maxdepth 1 -type d -name 'p133_moe_fixtures*' -o -name 'p133_fixtures*' | sort | tail -n 1 || true)"
fi
if [[ -z "$FIXTURE_DIR" || ! -f "$FIXTURE_DIR/manifest.json" ]]; then
  echo "ERROR: p133 fixture directory not found in $REPORT_DIR"
  echo ""
  echo "Run p133 first to export MoE fixtures:"
  echo "  python benchmarks/p133_export_active_moe_fixtures.py \\"
  echo "    --model $MODEL \\"
  echo "    --layers 0,4,8,16,20,28,32,36,39 \\"
  echo "    --out ${REPORT_DIR}/p133_moe_fixtures_official_w4a16"
  echo ""
  echo "Or set FIXTURE_DIR=/path/to/p133_fixtures"
  exit 1
fi

echo "[p189] root=$ROOT"
echo "[p189] model=$MODEL"
echo "[p189] fixture_dir=$FIXTURE_DIR"
echo "[p189] fp8_format=$FP8_FORMAT granularity=$GRANULARITY"
echo "[p189] out=$P189_JSON"

# ── Run P189 gate ───────────────────────────────────────────────────────────
"$PYTHON_BIN" benchmarks/p189_qwen36_35b_moe_w4a8_route_flip_gate.py \
  --fixtures "$FIXTURE_DIR" \
  --model "$MODEL" \
  --out "$P189_JSON" \
  --device "$DEVICE" \
  --fp8-format "$FP8_FORMAT" \
  --granularity "$GRANULARITY"

echo "[p189] done: $P189_JSON"
