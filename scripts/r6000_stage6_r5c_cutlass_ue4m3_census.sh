#!/usr/bin/env bash
# AutoDL/R6000 wrapper for the Stage 6 R5-C CUTLASS UE4M3 ABI census.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CUDA_BIN="${CUDA_BIN:-/usr/local/cuda-12.8/bin}"
CUTLASS_DIR="${CUTLASS_DIR:-/root/autodl-tmp/src/cutlass}"

export PATH="/root/miniconda3/bin:${CUDA_BIN}:${PATH}"
export TMPDIR="${TMPDIR:-${DATA_ROOT}/.tmp}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"
export LYNN_NATIVE_CUDA_ARCH="${LYNN_NATIVE_CUDA_ARCH:-sm_120a}"

mkdir -p "$TMPDIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT}/reports/stage6/r5c_cutlass_ue4m3_census_${STAMP}}"
RESULT_JSON="${OUT_DIR}/result.json"
SUMMARY_MD="${OUT_DIR}/summary.md"
mkdir -p "$OUT_DIR"

cd "$ROOT"
echo "[r5c-cutlass] root=$ROOT"
echo "[r5c-cutlass] cutlass_dir=$CUTLASS_DIR"
echo "[r5c-cutlass] out_dir=$OUT_DIR"
nvidia-smi >"${OUT_DIR}/nvidia_smi_before.txt" || true
git rev-parse HEAD >"${OUT_DIR}/git_head.txt"
git status --short >"${OUT_DIR}/git_status.txt"

set +e
"$PYTHON_BIN" scripts/r6000_stage6_r5c_cutlass_ue4m3_census.py \
  --cutlass-dir "$CUTLASS_DIR" \
  --out "$RESULT_JSON" \
  "$@" | tee "${OUT_DIR}/run.log"
probe_rc=${PIPESTATUS[0]}
set -e

summary_rc=2
if [[ -f "$RESULT_JSON" ]]; then
  set +e
  "$PYTHON_BIN" scripts/summarize_stage6_r5c_cutlass_ue4m3_census.py \
    "$RESULT_JSON" \
    --markdown-out "$SUMMARY_MD" \
    --strict-exit
  summary_rc=$?
  set -e
else
  echo "[r5c-cutlass] missing result json after probe failure" | tee "$SUMMARY_MD"
fi

nvidia-smi >"${OUT_DIR}/nvidia_smi_after.txt" || true
printf "%s\n" "$probe_rc" >"${OUT_DIR}/probe_exit_code.txt"
printf "%s\n" "$summary_rc" >"${OUT_DIR}/summary_exit_code.txt"
echo "[r5c-cutlass] result=$RESULT_JSON"
echo "[r5c-cutlass] summary=$SUMMARY_MD"

if [[ "$probe_rc" != "0" ]]; then
  exit "$probe_rc"
fi
exit "$summary_rc"
