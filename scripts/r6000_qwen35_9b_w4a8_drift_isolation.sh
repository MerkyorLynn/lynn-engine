#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# R6000: P197b W4A8 drift source isolation probe
#
# Runs the P197b isolation probe to determine whether token drift
# in the true FP4xFP8 resident path comes from:
#   - MMA fragment layout bug (P191)
#   - FP8 quantization error in gate/up activation
#   - FP8 re-quantization of silu*up intermediate for down_proj
#   - Compound error from all projections
#
# Requires:
#   - FP4xFP8 sidecar tensors at SIDECAR_DIR
#   - Lynn native CUDA extension (auto-builds)
#   - R6000 eval environment
# ─────────────────────────────────────────────────────────────

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MAX_NEW="${MAX_NEW:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
LIMIT="${LIMIT:-5}"
MODES="${MODES:-scalar_full,scalar_gateup_only,scalar_down_only,mma_full}"

# Ensure CUDA arch for native extension build
export LYNN_NATIVE_CUDA_ARCH="${LYNN_NATIVE_CUDA_ARCH:-sm_120a}"
export LYNN_ENABLE_SM120A_FP4_MMA="${LYNN_ENABLE_SM120A_FP4_MMA:-1}"

# Output paths
OUT_JSON="${REPORT_DIR}/p197b_w4a8_drift_isolation_${STAMP}.json"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

echo "[p197b] Model: $MODEL"
echo "[p197b] Modes: $MODES"
echo "[p197b] Output: $OUT_JSON"
echo ""

exec "$PYTHON_BIN" benchmarks/p197b_qwen35_9b_w4a8_drift_isolation.py \
  --model "$MODEL" \
  --max-new "$MAX_NEW" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --limit "$LIMIT" \
  --modes "$MODES" \
  --out "$OUT_JSON"
