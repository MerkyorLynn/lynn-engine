#!/usr/bin/env bash
# Run Stage 6 P4B runtime bridge fail-loud preflight on Spark and pull artifacts.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_spark_stage6_p4b_runtime_bridge_preflight.sh [options]

Options:
  --host HOST                  SSH host alias. Default: $LYNN_SPARK_HOST or dgx-spark.
  --image IMAGE                Docker image. Default: lynn-eval-base:cu13.
  --remote-repo PATH           Spark repo path. Default: /home/merkyor/lynn-engine.
  --model PATH                 Model path on Spark.
  --local-root PATH            Local artifact root. Default: reports/stage6.
  --expect-head COMMIT         Expected Spark repo HEAD. Default: local HEAD.
  --layer N                    Layer index. Default: 0.
  --allow-provenance-mismatch  Do not fail when both HEAD and manifest differ.
  --no-strict                  Pull artifacts but do not fail on result passes.all=false.
  -h, --help                   Show this help.

Environment overrides:
  LYNN_SPARK_HOST
  LYNN_SPARK_IMAGE
  LYNN_SPARK_REPO
  LYNN_STAGE6_LOCAL_OUT
  LYNN_STAGE6_EXPECT_HEAD
  LYNN_STAGE6_EXPECT_MANIFEST
  LYNN_STAGE6_P4_MODEL
USAGE
}

shell_quote() {
  printf '%q' "$1"
}

file_sha256() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

manifest_for_files() {
  local file
  for file in "$@"; do
    if [[ ! -f "$file" ]]; then
      echo "missing $file"
      continue
    fi
    printf '%s %s\n' "$(file_sha256 "$file")" "$file"
  done
}

PROVENANCE_FILES=(
  "scripts/run_spark_stage6_p4b_runtime_bridge_preflight.sh"
  "scripts/spark_stage6_p4b_runtime_bridge_preflight.py"
  "scripts/summarize_stage6_p4b_runtime_bridge_preflight.py"
  "scripts/test_stage6_p4b_runtime_bridge_tools.py"
  "scripts/test_stage6_p4b_single_kernel_static.py"
  "engine/native_cuda.py"
  "engine/moe_packed_nvfp4.py"
  "engine/resident_runner.py"
  "csrc/lynn_native/bindings.cpp"
  "csrc/lynn_native/moe_fused_zero_shadow_contract.cu"
  "reports/stage6/P4B_NATIVE_FUSED_SINGLE_KERNEL_CONTRACT_20260604.md"
)

HOST="${LYNN_SPARK_HOST:-dgx-spark}"
IMAGE="${LYNN_SPARK_IMAGE:-lynn-eval-base:cu13}"
REMOTE_REPO="${LYNN_SPARK_REPO:-/home/merkyor/lynn-engine}"
MODEL="${LYNN_STAGE6_P4_MODEL:-/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526}"
LOCAL_ROOT="${LYNN_STAGE6_LOCAL_OUT:-reports/stage6}"
EXPECTED_HEAD="${LYNN_STAGE6_EXPECT_HEAD:-$(git rev-parse HEAD 2>/dev/null || true)}"
EXPECTED_MANIFEST="${LYNN_STAGE6_EXPECT_MANIFEST:-$(manifest_for_files "${PROVENANCE_FILES[@]}")}"
LAYER="0"
REQUIRE_PROVENANCE="1"
STRICT="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --remote-repo) REMOTE_REPO="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --local-root) LOCAL_ROOT="$2"; shift 2 ;;
    --expect-head) EXPECTED_HEAD="$2"; shift 2 ;;
    --layer) LAYER="$2"; shift 2 ;;
    --allow-provenance-mismatch) REQUIRE_PROVENANCE="0"; shift ;;
    --no-strict) STRICT="0"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="p4b_runtime_bridge_preflight_${STAMP}"
REMOTE_RUN_DIR="${REMOTE_REPO}/reports/stage6/${RUN_NAME}"
LOCAL_RUN_DIR="${LOCAL_ROOT}/${RUN_NAME}"

echo "[p4b-runtime] host=${HOST}"
echo "[p4b-runtime] expect_head=${EXPECTED_HEAD:-none}"
echo "[p4b-runtime] remote_run_dir=${REMOTE_RUN_DIR}"

set +e
ssh "$HOST" \
  "REMOTE_REPO=$(shell_quote "$REMOTE_REPO") REMOTE_RUN_DIR=$(shell_quote "$REMOTE_RUN_DIR") IMAGE=$(shell_quote "$IMAGE") MODEL=$(shell_quote "$MODEL") LAYER=$(shell_quote "$LAYER") EXPECTED_HEAD=$(shell_quote "$EXPECTED_HEAD") EXPECTED_MANIFEST=$(shell_quote "$EXPECTED_MANIFEST") REQUIRE_PROVENANCE=$(shell_quote "$REQUIRE_PROVENANCE") STRICT=$(shell_quote "$STRICT") bash -s" <<'REMOTE'
set -euo pipefail
file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

manifest_for_files() {
  local file
  for file in "$@"; do
    if [[ ! -f "$file" ]]; then
      echo "missing $file"
      continue
    fi
    printf '%s %s\n' "$(file_sha256 "$file")" "$file"
  done
}

PROVENANCE_FILES=(
  "scripts/run_spark_stage6_p4b_runtime_bridge_preflight.sh"
  "scripts/spark_stage6_p4b_runtime_bridge_preflight.py"
  "scripts/summarize_stage6_p4b_runtime_bridge_preflight.py"
  "scripts/test_stage6_p4b_runtime_bridge_tools.py"
  "scripts/test_stage6_p4b_single_kernel_static.py"
  "engine/native_cuda.py"
  "engine/moe_packed_nvfp4.py"
  "engine/resident_runner.py"
  "csrc/lynn_native/bindings.cpp"
  "csrc/lynn_native/moe_fused_zero_shadow_contract.cu"
  "reports/stage6/P4B_NATIVE_FUSED_SINGLE_KERNEL_CONTRACT_20260604.md"
)

cd "$REMOTE_REPO"
mkdir -p "$REMOTE_RUN_DIR"
REMOTE_HEAD="$(git rev-parse HEAD 2>/dev/null || true)"
REMOTE_MANIFEST="$(manifest_for_files "${PROVENANCE_FILES[@]}")"
printf '%s\n' "$REMOTE_HEAD" > "$REMOTE_RUN_DIR/git_head.txt"
printf '%s\n' "${EXPECTED_HEAD:-}" > "$REMOTE_RUN_DIR/expected_git_head.txt"
printf '%s\n' "$REMOTE_MANIFEST" > "$REMOTE_RUN_DIR/provenance_manifest.txt"
printf '%s\n' "$EXPECTED_MANIFEST" > "$REMOTE_RUN_DIR/expected_provenance_manifest.txt"
git status --short > "$REMOTE_RUN_DIR/git_status.txt" 2>/dev/null || true
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader > "$REMOTE_RUN_DIR/nvidia_smi_before.txt" 2>/dev/null || true

if [[ "$REMOTE_HEAD" == "${EXPECTED_HEAD:-}" ]]; then
  echo "remote HEAD ok" > "$REMOTE_RUN_DIR/head_check.txt"
elif [[ "$REMOTE_MANIFEST" == "${EXPECTED_MANIFEST:-}" ]]; then
  {
    echo "remote manifest ok"
    echo "expected HEAD: ${EXPECTED_HEAD:-}"
    echo "actual HEAD:   $REMOTE_HEAD"
  } > "$REMOTE_RUN_DIR/head_check.txt"
elif [[ "${REQUIRE_PROVENANCE:-1}" == "1" ]]; then
  {
    echo "remote provenance mismatch"
    echo "expected HEAD: ${EXPECTED_HEAD:-}"
    echo "actual HEAD:   $REMOTE_HEAD"
    echo "expected manifest:"
    echo "$EXPECTED_MANIFEST"
    echo "actual manifest:"
    echo "$REMOTE_MANIFEST"
  } > "$REMOTE_RUN_DIR/head_check.txt"
  exit 12
else
  echo "remote provenance mismatch allowed" > "$REMOTE_RUN_DIR/head_check.txt"
fi

STRICT_FLAG="--strict-exit"
if [[ "${STRICT:-1}" != "1" ]]; then
  STRICT_FLAG=""
fi

set +e
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 \
  -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w "$REMOTE_REPO" \
  "$IMAGE" \
  python3 scripts/spark_stage6_p4b_runtime_bridge_preflight.py \
    --model "$MODEL" \
    --out "$REMOTE_RUN_DIR/result.json" \
    --layer "$LAYER" \
    $STRICT_FLAG \
  > "$REMOTE_RUN_DIR/run.log" 2>&1
DOCKER_STATUS=$?
set -e
printf '%s\n' "$DOCKER_STATUS" > "$REMOTE_RUN_DIR/docker_exit_code.txt"
if [[ -f "$REMOTE_RUN_DIR/result.json" ]]; then
  python3 scripts/summarize_stage6_p4b_runtime_bridge_preflight.py \
    "$REMOTE_RUN_DIR/result.json" \
    --markdown-out "$REMOTE_RUN_DIR/summary.md" \
    $STRICT_FLAG \
    >> "$REMOTE_RUN_DIR/run.log" 2>&1
  SUMMARY_STATUS=$?
  if [[ "$DOCKER_STATUS" -eq 0 ]]; then
    DOCKER_STATUS="$SUMMARY_STATUS"
  fi
fi
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader > "$REMOTE_RUN_DIR/nvidia_smi_after.txt" 2>/dev/null || true
exit "$DOCKER_STATUS"
REMOTE
REMOTE_STATUS=$?
set -e

mkdir -p "$LOCAL_RUN_DIR"
set +e
rsync -av "${HOST}:${REMOTE_RUN_DIR}/" "$LOCAL_RUN_DIR/"
RSYNC_STATUS=$?
set -e
if [[ "$RSYNC_STATUS" -ne 0 ]]; then
  echo "[p4b-runtime] rsync failed with status ${RSYNC_STATUS}" >&2
  exit "$RSYNC_STATUS"
fi

if [[ ! -f "$LOCAL_RUN_DIR/result.json" ]]; then
  echo "[p4b-runtime] missing result.json; pulled failure artifacts to ${LOCAL_RUN_DIR}" >&2
  if [[ -f "$LOCAL_RUN_DIR/head_check.txt" ]]; then
    cat "$LOCAL_RUN_DIR/head_check.txt" >&2 || true
  fi
  if [[ -f "$LOCAL_RUN_DIR/run.log" ]]; then
    tail -n 80 "$LOCAL_RUN_DIR/run.log" >&2 || true
  fi
  exit "$REMOTE_STATUS"
fi

LOCAL_STRICT_FLAG="--strict-exit"
if [[ "$STRICT" != "1" ]]; then
  LOCAL_STRICT_FLAG=""
fi
python3 scripts/summarize_stage6_p4b_runtime_bridge_preflight.py \
  "$LOCAL_RUN_DIR/result.json" \
  --markdown-out "$LOCAL_RUN_DIR/summary.md" \
  $LOCAL_STRICT_FLAG

echo "[p4b-runtime] artifacts: ${LOCAL_RUN_DIR}"
if [[ "$REMOTE_STATUS" -ne 0 && "$STRICT" == "1" ]]; then
  exit "$REMOTE_STATUS"
fi
