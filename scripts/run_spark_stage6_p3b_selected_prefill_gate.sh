#!/usr/bin/env bash
# Run Stage 6 P3-B selected-prefill composition gate on Spark and pull artifacts.
#
# Example:
#   scripts/run_spark_stage6_p3b_selected_prefill_gate.sh --layers 0-3 --tokens 16,64 --predecessors-pass
#   scripts/run_spark_stage6_p3b_selected_prefill_gate.sh --host dgx-via-ssh --no-strict
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_spark_stage6_p3b_selected_prefill_gate.sh [options]

Options:
  --host HOST                  SSH host alias. Default: $LYNN_SPARK_HOST or dgx-spark.
  --model PATH                 Model path on Spark.
  --layers SPEC                Selected layer spec. Default: 0-3.
  --tokens CSV                 Sequence lengths/tokens. Default: 16,64.
  --image IMAGE                Docker image. Default: lynn-eval-base:cu13.
  --remote-repo PATH           Spark repo path. Default: /home/merkyor/lynn-engine.
  --local-root PATH            Local artifact root. Default: reports/stage6.
  --expect-head COMMIT         Expected Spark repo HEAD. Default: local HEAD.
  --predecessors-pass          Assert P2-O basic, P2-O rc-mini, P3-A, and suite predecessors passed.
  --allow-provenance-mismatch  Do not fail when both HEAD and manifest differ.
  --no-strict                  Pull artifacts but do not fail on result passes.all=false.
  -h, --help                   Show this help.

Environment overrides:
  LYNN_SPARK_HOST
  LYNN_STAGE6_MODEL
  LYNN_SPARK_IMAGE
  LYNN_SPARK_REPO
  LYNN_STAGE6_LOCAL_OUT
  LYNN_STAGE6_EXPECT_HEAD
  LYNN_STAGE6_EXPECT_MANIFEST
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
  "scripts/run_spark_stage6_p3b_selected_prefill_gate.sh"
  "scripts/spark_stage6_p3b_selected_prefill_gate.py"
  "scripts/summarize_stage6_p3b_selected_prefill_gate.py"
  "scripts/write_stage6_p3b_report.py"
  "engine/full_forward.py"
  "engine/moe_packed_nvfp4.py"
  "triton_kernels/nvfp4_moe.py"
  "reports/stage6/P3B_SELECTED_PREFILL_GATE_RUNBOOK_20260604.md"
)

HOST="${LYNN_SPARK_HOST:-dgx-spark}"
MODEL="${LYNN_STAGE6_MODEL:-/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526}"
LAYERS="0-3"
TOKENS="16,64"
IMAGE="${LYNN_SPARK_IMAGE:-lynn-eval-base:cu13}"
REMOTE_REPO="${LYNN_SPARK_REPO:-/home/merkyor/lynn-engine}"
LOCAL_ROOT="${LYNN_STAGE6_LOCAL_OUT:-reports/stage6}"
EXPECTED_HEAD="${LYNN_STAGE6_EXPECT_HEAD:-$(git rev-parse HEAD 2>/dev/null || true)}"
EXPECTED_MANIFEST="${LYNN_STAGE6_EXPECT_MANIFEST:-$(manifest_for_files "${PROVENANCE_FILES[@]}")}"
REQUIRE_PROVENANCE="1"
STRICT="1"
PREDECESSORS_PASS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --layers)
      LAYERS="$2"
      shift 2
      ;;
    --tokens|--seq-lens)
      TOKENS="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --remote-repo)
      REMOTE_REPO="$2"
      shift 2
      ;;
    --local-root)
      LOCAL_ROOT="$2"
      shift 2
      ;;
    --expect-head)
      EXPECTED_HEAD="$2"
      shift 2
      ;;
    --predecessors-pass)
      PREDECESSORS_PASS="1"
      shift
      ;;
    --allow-provenance-mismatch)
      REQUIRE_PROVENANCE="0"
      shift
      ;;
    --no-strict)
      STRICT="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="p3b_layers${LAYERS//[^0-9A-Za-z_-]/_}_selected_prefill_gate_${STAMP}"
REMOTE_RUN_DIR="${REMOTE_REPO}/reports/stage6/${RUN_NAME}"
LOCAL_RUN_DIR="${LOCAL_ROOT}/${RUN_NAME}"

echo "[p3b] host=${HOST}"
echo "[p3b] layers=${LAYERS}"
echo "[p3b] tokens=${TOKENS}"
echo "[p3b] predecessors_pass=${PREDECESSORS_PASS}"
echo "[p3b] expect_head=${EXPECTED_HEAD:-none}"
echo "[p3b] remote_run_dir=${REMOTE_RUN_DIR}"

set +e
ssh "$HOST" \
  "REMOTE_REPO=$(shell_quote "$REMOTE_REPO") REMOTE_RUN_DIR=$(shell_quote "$REMOTE_RUN_DIR") IMAGE=$(shell_quote "$IMAGE") MODEL=$(shell_quote "$MODEL") LAYERS=$(shell_quote "$LAYERS") TOKENS=$(shell_quote "$TOKENS") EXPECTED_HEAD=$(shell_quote "$EXPECTED_HEAD") EXPECTED_MANIFEST=$(shell_quote "$EXPECTED_MANIFEST") REQUIRE_PROVENANCE=$(shell_quote "$REQUIRE_PROVENANCE") PREDECESSORS_PASS=$(shell_quote "$PREDECESSORS_PASS") bash -s" <<'REMOTE'
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
  "scripts/run_spark_stage6_p3b_selected_prefill_gate.sh"
  "scripts/spark_stage6_p3b_selected_prefill_gate.py"
  "scripts/summarize_stage6_p3b_selected_prefill_gate.py"
  "scripts/write_stage6_p3b_report.py"
  "engine/full_forward.py"
  "engine/moe_packed_nvfp4.py"
  "triton_kernels/nvfp4_moe.py"
  "reports/stage6/P3B_SELECTED_PREFILL_GATE_RUNBOOK_20260604.md"
)

cd "$REMOTE_REPO"
PRE_RUN_GIT_STATUS="$(git status --short 2>/dev/null || true)"
mkdir -p "$REMOTE_RUN_DIR"
REMOTE_HEAD="$(git rev-parse HEAD 2>/dev/null || true)"
REMOTE_MANIFEST="$(manifest_for_files "${PROVENANCE_FILES[@]}")"
printf '%s\n' "$REMOTE_HEAD" > "$REMOTE_RUN_DIR/git_head.txt"
printf '%s\n' "${EXPECTED_HEAD:-}" > "$REMOTE_RUN_DIR/expected_git_head.txt"
printf '%s\n' "$REMOTE_MANIFEST" > "$REMOTE_RUN_DIR/provenance_manifest.txt"
printf '%s\n' "$EXPECTED_MANIFEST" > "$REMOTE_RUN_DIR/expected_provenance_manifest.txt"
printf '%s\n' "$PRE_RUN_GIT_STATUS" > "$REMOTE_RUN_DIR/git_status.txt"
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
  {
    echo "remote provenance mismatch allowed"
    echo "expected HEAD: ${EXPECTED_HEAD:-}"
    echo "actual HEAD:   $REMOTE_HEAD"
  } > "$REMOTE_RUN_DIR/head_check.txt"
fi

PRED_ARG=()
if [[ "${PREDECESSORS_PASS:-0}" == "1" ]]; then
  PRED_ARG=(--predecessors-pass)
fi

set +e
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 \
  -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w "$REMOTE_REPO" \
  "$IMAGE" \
  python3 scripts/spark_stage6_p3b_selected_prefill_gate.py \
    --model "$MODEL" \
    --layers "$LAYERS" \
    --tokens "$TOKENS" \
    "${PRED_ARG[@]}" \
    --json-out "$REMOTE_RUN_DIR/result.json" \
  > "$REMOTE_RUN_DIR/run.log" 2>&1
DOCKER_STATUS=$?
set -e
printf '%s\n' "$DOCKER_STATUS" > "$REMOTE_RUN_DIR/docker_exit_code.txt"
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
  echo "[p3b] rsync failed with status ${RSYNC_STATUS}" >&2
  exit "$RSYNC_STATUS"
fi

if [[ ! -f "$LOCAL_RUN_DIR/result.json" ]]; then
  echo "[p3b] missing result.json; pulled failure artifacts to ${LOCAL_RUN_DIR}" >&2
  if [[ -f "$LOCAL_RUN_DIR/head_check.txt" ]]; then
    cat "$LOCAL_RUN_DIR/head_check.txt" >&2 || true
  fi
  if [[ -f "$LOCAL_RUN_DIR/run.log" ]]; then
    tail -n 80 "$LOCAL_RUN_DIR/run.log" >&2 || true
  fi
  {
    echo "# Stage 6 P3-B Selected-Prefill Gate Summary"
    echo
    echo "| Field | Value |"
    echo "|---|---|"
    echo "| Verdict | **FAIL** (missing result.json) |"
    echo "| Artifact | \`${LOCAL_RUN_DIR}\` |"
    echo "| Docker exit code | \`$(cat "$LOCAL_RUN_DIR/docker_exit_code.txt" 2>/dev/null || echo missing)\` |"
    echo "| Head check | \`$(tr '\n' ' ' < "$LOCAL_RUN_DIR/head_check.txt" 2>/dev/null || echo missing)\` |"
    echo
    echo "P3-B could not produce a JSON artifact. Do not bank this run."
  } > "$LOCAL_RUN_DIR/summary.md"
  {
    echo "# Stage 6 Phase 3-B — selected-prefill composition gate"
    echo
    echo "Verdict: **FAIL** (missing result.json)."
    echo
    echo "Artifact directory: \`${LOCAL_RUN_DIR}\`"
    echo
    echo "This failure report was generated by the wrapper after pulling remote"
    echo "artifacts that did not include \`result.json\`. It may indicate provenance,"
    echo "import, CUDA, OOM, or early process failure."
    echo
    echo "## Head Check"
    echo
    echo '```text'
    cat "$LOCAL_RUN_DIR/head_check.txt" 2>/dev/null || true
    echo '```'
    echo
    echo "## Run Log Tail"
    echo
    echo '```text'
    tail -n 100 "$LOCAL_RUN_DIR/run.log" 2>/dev/null || true
    echo '```'
    echo
    echo "## Decision"
    echo
    echo "Do not bank P3-B. Re-run after fixing the missing JSON failure."
  } > "$LOCAL_RUN_DIR/report.md"
  if [[ "$REMOTE_STATUS" -ne 0 ]]; then
    exit "$REMOTE_STATUS"
  fi
  exit 3
fi

SUMMARY_ARGS=("$LOCAL_RUN_DIR/result.json" "--markdown-out" "$LOCAL_RUN_DIR/summary.md")
if [[ "$STRICT" == "1" ]]; then
  SUMMARY_ARGS+=("--strict-exit")
fi
set +e
python3 scripts/summarize_stage6_p3b_selected_prefill_gate.py "${SUMMARY_ARGS[@]}"
SUMMARY_STATUS=$?
set -e

python3 scripts/write_stage6_p3b_report.py \
  "$LOCAL_RUN_DIR" \
  --report-out "$LOCAL_RUN_DIR/report.md" || true

echo "[p3b] artifacts: ${LOCAL_RUN_DIR}"
if [[ "$REMOTE_STATUS" -ne 0 && "$STRICT" == "1" ]]; then
  exit "$REMOTE_STATUS"
fi
exit "$SUMMARY_STATUS"
