#!/usr/bin/env bash
# Run Stage 6 P3-E RC quality-battery smoke on Spark and pull artifacts.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_spark_stage6_p3e_rc_quality_battery.sh [options]

Options:
  --host HOST                  SSH host alias. Default: $LYNN_SPARK_HOST or dgx-spark.
  --model PATH                 Model path on Spark.
  --max-seq-len N              Server max_seq_len. Default: 32768.
  --port PORT                  Server port inside the container. Default: 18371.
  --mmlu-data-dir PATH         MMLU CSV directory on Spark.
  --gpqa-csv PATH              GPQA Diamond CSV on Spark.
  --mmlu-sample N              MMLU sample count. Default: 100.
  --gpqa-sample N              GPQA sample count. Default: 50.
  --mmlu-floor FLOAT           MMLU sample accuracy floor. Default: 0.70.
  --gpqa-floor FLOAT           GPQA sample accuracy floor. Default: 0.30.
  --longctx-target-tokens N    Long-context needle target. Default: 8192.
  --image IMAGE                Docker image. Default: lynn-eval-base:cu13.
  --remote-repo PATH           Spark repo path. Default: /home/merkyor/lynn-engine.
  --local-root PATH            Local artifact root. Default: reports/stage6.
  --expect-head COMMIT         Expected Spark repo HEAD. Default: local HEAD.
  --p3d-pass                   Assert P3-D server-smoke predecessor passed.
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
  "scripts/run_spark_stage6_p3e_rc_quality_battery.sh"
  "scripts/spark_stage6_p3e_rc_quality_battery.py"
  "scripts/summarize_stage6_p3e_rc_quality_battery.py"
  "scripts/write_stage6_p3e_report.py"
  "scripts/openai_mmlu_500_5shot_eval.py"
  "scripts/openai_gpqa_diamond_eval.py"
  "scripts/spark_stage6_p3d_server_rc_gate.py"
  "server/openai_http.py"
  "engine/full_forward.py"
  "engine/resident_runner.py"
  "reports/stage6/P3E_RC_QUALITY_BATTERY_RUNBOOK_20260604.md"
)

HOST="${LYNN_SPARK_HOST:-dgx-spark}"
MODEL="${LYNN_STAGE6_MODEL:-/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526}"
MAX_SEQ_LEN="32768"
PORT="18371"
MMLU_DATA_DIR="/home/merkyor/lynn-nemotron-eval/mmlu_csv"
GPQA_CSV="/home/merkyor/quality-eval-20260517/datasets/gpqa/gpqa_diamond.csv"
MMLU_SAMPLE="100"
GPQA_SAMPLE="50"
MMLU_FLOOR="0.70"
GPQA_FLOOR="0.30"
LONGCTX_TARGET_TOKENS="8192"
IMAGE="${LYNN_SPARK_IMAGE:-lynn-eval-base:cu13}"
REMOTE_REPO="${LYNN_SPARK_REPO:-/home/merkyor/lynn-engine}"
LOCAL_ROOT="${LYNN_STAGE6_LOCAL_OUT:-reports/stage6}"
EXPECTED_HEAD="${LYNN_STAGE6_EXPECT_HEAD:-$(git rev-parse HEAD 2>/dev/null || true)}"
EXPECTED_MANIFEST="${LYNN_STAGE6_EXPECT_MANIFEST:-$(manifest_for_files "${PROVENANCE_FILES[@]}")}"
REQUIRE_PROVENANCE="1"
STRICT="1"
P3D_PASS="0"

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
    --max-seq-len)
      MAX_SEQ_LEN="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --mmlu-data-dir)
      MMLU_DATA_DIR="$2"
      shift 2
      ;;
    --gpqa-csv)
      GPQA_CSV="$2"
      shift 2
      ;;
    --mmlu-sample)
      MMLU_SAMPLE="$2"
      shift 2
      ;;
    --gpqa-sample)
      GPQA_SAMPLE="$2"
      shift 2
      ;;
    --mmlu-floor)
      MMLU_FLOOR="$2"
      shift 2
      ;;
    --gpqa-floor)
      GPQA_FLOOR="$2"
      shift 2
      ;;
    --longctx-target-tokens)
      LONGCTX_TARGET_TOKENS="$2"
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
    --p3d-pass)
      P3D_PASS="1"
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
RUN_NAME="p3e_rc_quality_battery_${STAMP}"
REMOTE_RUN_DIR="${REMOTE_REPO}/reports/stage6/${RUN_NAME}"
LOCAL_RUN_DIR="${LOCAL_ROOT}/${RUN_NAME}"

echo "[p3e] host=${HOST}"
echo "[p3e] p3d_pass=${P3D_PASS}"
echo "[p3e] expect_head=${EXPECTED_HEAD:-none}"
echo "[p3e] remote_run_dir=${REMOTE_RUN_DIR}"

set +e
ssh "$HOST" \
  "REMOTE_REPO=$(shell_quote "$REMOTE_REPO") REMOTE_RUN_DIR=$(shell_quote "$REMOTE_RUN_DIR") IMAGE=$(shell_quote "$IMAGE") MODEL=$(shell_quote "$MODEL") MAX_SEQ_LEN=$(shell_quote "$MAX_SEQ_LEN") PORT=$(shell_quote "$PORT") MMLU_DATA_DIR=$(shell_quote "$MMLU_DATA_DIR") GPQA_CSV=$(shell_quote "$GPQA_CSV") MMLU_SAMPLE=$(shell_quote "$MMLU_SAMPLE") GPQA_SAMPLE=$(shell_quote "$GPQA_SAMPLE") MMLU_FLOOR=$(shell_quote "$MMLU_FLOOR") GPQA_FLOOR=$(shell_quote "$GPQA_FLOOR") LONGCTX_TARGET_TOKENS=$(shell_quote "$LONGCTX_TARGET_TOKENS") EXPECTED_HEAD=$(shell_quote "$EXPECTED_HEAD") EXPECTED_MANIFEST=$(shell_quote "$EXPECTED_MANIFEST") REQUIRE_PROVENANCE=$(shell_quote "$REQUIRE_PROVENANCE") P3D_PASS=$(shell_quote "$P3D_PASS") bash -s" <<'REMOTE'
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
  "scripts/run_spark_stage6_p3e_rc_quality_battery.sh"
  "scripts/spark_stage6_p3e_rc_quality_battery.py"
  "scripts/summarize_stage6_p3e_rc_quality_battery.py"
  "scripts/write_stage6_p3e_report.py"
  "scripts/openai_mmlu_500_5shot_eval.py"
  "scripts/openai_gpqa_diamond_eval.py"
  "scripts/spark_stage6_p3d_server_rc_gate.py"
  "server/openai_http.py"
  "engine/full_forward.py"
  "engine/resident_runner.py"
  "reports/stage6/P3E_RC_QUALITY_BATTERY_RUNBOOK_20260604.md"
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

P3D_ARG=()
if [[ "${P3D_PASS:-0}" == "1" ]]; then
  P3D_ARG=(--p3d-pass)
fi

set +e
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 \
  -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w "$REMOTE_REPO" \
  "$IMAGE" \
  python3 scripts/spark_stage6_p3e_rc_quality_battery.py \
    --model "$MODEL" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --port "$PORT" \
    --mmlu-data-dir "$MMLU_DATA_DIR" \
    --gpqa-csv "$GPQA_CSV" \
    --mmlu-sample "$MMLU_SAMPLE" \
    --gpqa-sample "$GPQA_SAMPLE" \
    --mmlu-floor "$MMLU_FLOOR" \
    --gpqa-floor "$GPQA_FLOOR" \
    --longctx-target-tokens "$LONGCTX_TARGET_TOKENS" \
    --work-dir "$REMOTE_RUN_DIR" \
    "${P3D_ARG[@]}" \
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
  echo "[p3e] rsync failed with status ${RSYNC_STATUS}" >&2
  exit "$RSYNC_STATUS"
fi

if [[ ! -f "$LOCAL_RUN_DIR/result.json" ]]; then
  echo "[p3e] missing result.json; pulled failure artifacts to ${LOCAL_RUN_DIR}" >&2
  {
    echo "# Stage 6 P3-E RC Quality-Battery Summary"
    echo
    echo "| Field | Value |"
    echo "|---|---|"
    echo "| Verdict | **FAIL** (missing result.json) |"
    echo "| Artifact | \`${LOCAL_RUN_DIR}\` |"
    echo "| Docker exit code | \`$(cat "$LOCAL_RUN_DIR/docker_exit_code.txt" 2>/dev/null || echo missing)\` |"
    echo "| Head check | \`$(tr '\n' ' ' < "$LOCAL_RUN_DIR/head_check.txt" 2>/dev/null || echo missing)\` |"
    echo
    echo "P3-E could not produce a JSON artifact. Do not bank this run."
  } > "$LOCAL_RUN_DIR/summary.md"
  {
    echo "# Stage 6 Phase 3-E — RC quality-battery smoke"
    echo
    echo "Verdict: **FAIL** (missing result.json)."
    echo
    echo "Artifact directory: \`${LOCAL_RUN_DIR}\`"
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
    echo "Do not bank P3-E. Re-run after fixing the missing JSON failure."
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
python3 scripts/summarize_stage6_p3e_rc_quality_battery.py "${SUMMARY_ARGS[@]}"
SUMMARY_STATUS=$?
set -e

python3 scripts/write_stage6_p3e_report.py \
  "$LOCAL_RUN_DIR" \
  --report-out "$LOCAL_RUN_DIR/report.md" || true

echo "[p3e] artifacts: ${LOCAL_RUN_DIR}"
if [[ "$REMOTE_STATUS" -ne 0 && "$STRICT" == "1" ]]; then
  exit "$REMOTE_STATUS"
fi
exit "$SUMMARY_STATUS"
