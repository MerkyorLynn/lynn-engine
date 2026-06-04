#!/usr/bin/env bash
# AutoDL RTX PRO 6000 Stage6 FP4-MMA bring-up wrapper.
#
# Conservative by default: it does not install large packages such as vLLM.
# It keeps caches/build outputs on /root/autodl-tmp so the 30G system disk stays
# clean, records machine inventory, runs the R6000 census gate, and writes a
# summary even when the census fails.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CUDA_BIN="${CUDA_BIN:-/usr/local/cuda-12.8/bin}"
INSTALL_BASICS=0
DRY_RUN=0
CUTLASS_DIR="${CUTLASS_DIR:-}"
MIN_DISK_GIB="${MIN_DISK_GIB:-500}"

usage() {
  cat <<'EOF'
Usage: scripts/r6000_stage6_autodl_bootstrap.sh [options]

Options:
  --install-basics      Install small Python build prerequisites only (ninja).
  --cutlass-dir PATH    CUTLASS checkout root or include dir.
  --min-disk-gib N      Minimum workspace disk gate passed to census (default: 500).
  --dry-run             Print commands without executing them.
  -h, --help            Show this help.

Notes:
  - Run from the lynn-engine repo on the AutoDL RTX PRO 6000 host.
  - vLLM/Marlin/Machete is intentionally not installed by default; install it
    explicitly after the first census if the public-kernel map is the only fail.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-basics)
      INSTALL_BASICS=1
      shift
      ;;
    --cutlass-dir)
      CUTLASS_DIR="${2:?--cutlass-dir requires a path}"
      shift 2
      ;;
    --min-disk-gib)
      MIN_DISK_GIB="${2:?--min-disk-gib requires a number}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[r6000-bootstrap][error] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

run_shell() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    bash -lc "$*"
  fi
}

export PATH="/root/miniconda3/bin:${CUDA_BIN}:${PATH}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_ROOT}/.cache/pip}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${DATA_ROOT}/.cache/torch_extensions}"
export TMPDIR="${TMPDIR:-${DATA_ROOT}/.tmp}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"
export LYNN_NATIVE_CUDA_ARCH="${LYNN_NATIVE_CUDA_ARCH:-sm_120a}"

mkdir -p "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"

if [[ -n "$CUTLASS_DIR" ]]; then
  if [[ -d "$CUTLASS_DIR/include" ]]; then
    export LYNN_CUTLASS_INCLUDE_DIR="$CUTLASS_DIR/include"
  else
    export LYNN_CUTLASS_INCLUDE_DIR="$CUTLASS_DIR"
  fi
elif [[ -d "${DATA_ROOT}/src/cutlass/include" ]]; then
  export LYNN_CUTLASS_INCLUDE_DIR="${DATA_ROOT}/src/cutlass/include"
fi

echo "[r6000-bootstrap] root=$ROOT"
echo "[r6000-bootstrap] data_root=$DATA_ROOT"
echo "[r6000-bootstrap] python=$PYTHON_BIN"
echo "[r6000-bootstrap] cuda_home=$CUDA_HOME"
echo "[r6000-bootstrap] torch_cuda_arch_list=$TORCH_CUDA_ARCH_LIST"
echo "[r6000-bootstrap] lynn_native_cuda_arch=$LYNN_NATIVE_CUDA_ARCH"
echo "[r6000-bootstrap] cutlass_include=${LYNN_CUTLASS_INCLUDE_DIR:-<unset>}"

if [[ "$INSTALL_BASICS" == "1" ]]; then
  run "$PYTHON_BIN" -m pip install --upgrade --no-input ninja
fi

run "$PYTHON_BIN" - <<'PY'
import json
import shutil
import subprocess
import torch

props = torch.cuda.get_device_properties(0)
payload = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "memory_gib": props.total_memory / (1024 ** 3),
    "nvcc": shutil.which("nvcc"),
    "ninja": shutil.which("ninja"),
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
subprocess.run(["nvidia-smi"], check=False)
subprocess.run(["df", "-h", "/root/autodl-tmp"], check=False)
PY

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/reports/stage6/r6000_fp4_mma_census_${STAMP}"
RESULT_JSON="${OUT_DIR}/result.json"
SUMMARY_MD="${OUT_DIR}/summary.md"
RUN_LOG="${OUT_DIR}/run.log"
mkdir -p "$OUT_DIR"

echo "[r6000-bootstrap] out_dir=$OUT_DIR"
set +e
if [[ "$DRY_RUN" == "1" ]]; then
  echo "+ $PYTHON_BIN scripts/r6000_stage6_fp4_mma_census.py --run-contracts --min-disk-gib $MIN_DISK_GIB --out $RESULT_JSON"
  census_rc=0
else
  "$PYTHON_BIN" scripts/r6000_stage6_fp4_mma_census.py \
    --run-contracts \
    --min-disk-gib "$MIN_DISK_GIB" \
    --out "$RESULT_JSON" | tee "$RUN_LOG"
  census_rc=${PIPESTATUS[0]}
fi
set -e

if [[ "$DRY_RUN" == "1" ]]; then
  echo "+ $PYTHON_BIN scripts/summarize_stage6_r6000_fp4_mma_census.py $RESULT_JSON --markdown-out $SUMMARY_MD"
elif [[ ! -f scripts/summarize_stage6_r6000_fp4_mma_census.py ]]; then
  echo "[r6000-bootstrap][warn] summary script missing; keeping raw census result only" >&2
else
  "$PYTHON_BIN" scripts/summarize_stage6_r6000_fp4_mma_census.py \
    "$RESULT_JSON" \
    --markdown-out "$SUMMARY_MD" || true
fi

echo "[r6000-bootstrap] result=$RESULT_JSON"
echo "[r6000-bootstrap] summary=$SUMMARY_MD"
exit "$census_rc"
