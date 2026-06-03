#!/usr/bin/env bash
# Run Stage 6 P2-O packed-prefill RC smoke on Spark and pull artifacts locally.
#
# Example:
#   scripts/run_spark_stage6_p2o_rc_smoke.sh --preset basic
#   scripts/run_spark_stage6_p2o_rc_smoke.sh --preset rc-mini --host dgx-via-ssh
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_spark_stage6_p2o_rc_smoke.sh [options]

Options:
  --preset basic|rc-mini       Prompt preset to run. Default: basic.
  --host HOST                  SSH host alias. Default: $LYNN_SPARK_HOST or dgx-spark.
  --model PATH                 Model path on Spark.
  --max-new N                  Generated tokens per prompt. Default: 8.
  --max-seq-len N              Runner max_seq_len. Default: 2048.
  --image IMAGE                Docker image. Default: lynn-eval-base:cu13.
  --remote-repo PATH           Spark repo path. Default: /home/merkyor/lynn-engine.
  --local-root PATH            Local artifact root. Default: reports/stage6.
  --no-strict                  Do not fail the script when the summary verdict is not PASS.
  -h, --help                   Show this help.

Environment overrides:
  LYNN_SPARK_HOST
  LYNN_STAGE6_MODEL
  LYNN_SPARK_IMAGE
  LYNN_SPARK_REPO
  LYNN_STAGE6_LOCAL_OUT
USAGE
}

shell_quote() {
  printf '%q' "$1"
}

PRESET="basic"
HOST="${LYNN_SPARK_HOST:-dgx-spark}"
MODEL="${LYNN_STAGE6_MODEL:-/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526}"
MAX_NEW="8"
MAX_SEQ_LEN="2048"
IMAGE="${LYNN_SPARK_IMAGE:-lynn-eval-base:cu13}"
REMOTE_REPO="${LYNN_SPARK_REPO:-/home/merkyor/lynn-engine}"
LOCAL_ROOT="${LYNN_STAGE6_LOCAL_OUT:-reports/stage6}"
STRICT="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)
      PRESET="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --max-new)
      MAX_NEW="$2"
      shift 2
      ;;
    --max-seq-len)
      MAX_SEQ_LEN="$2"
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

if [[ "$PRESET" != "basic" && "$PRESET" != "rc-mini" ]]; then
  echo "--preset must be basic or rc-mini" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="p2o_${PRESET}_packed_prefill_rc_smoke_${STAMP}"
REMOTE_RUN_DIR="${REMOTE_REPO}/reports/stage6/${RUN_NAME}"
LOCAL_RUN_DIR="${LOCAL_ROOT}/${RUN_NAME}"

echo "[p2o] host=${HOST}"
echo "[p2o] preset=${PRESET}"
echo "[p2o] remote_run_dir=${REMOTE_RUN_DIR}"

ssh "$HOST" \
  "REMOTE_REPO=$(shell_quote "$REMOTE_REPO") REMOTE_RUN_DIR=$(shell_quote "$REMOTE_RUN_DIR") IMAGE=$(shell_quote "$IMAGE") MODEL=$(shell_quote "$MODEL") PRESET=$(shell_quote "$PRESET") MAX_NEW=$(shell_quote "$MAX_NEW") MAX_SEQ_LEN=$(shell_quote "$MAX_SEQ_LEN") bash -s" <<'REMOTE'
set -euo pipefail
cd "$REMOTE_REPO"
mkdir -p "$REMOTE_RUN_DIR"
git rev-parse HEAD > "$REMOTE_RUN_DIR/git_head.txt" 2>/dev/null || true
git status --short > "$REMOTE_RUN_DIR/git_status.txt" 2>/dev/null || true
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader > "$REMOTE_RUN_DIR/nvidia_smi_before.txt" 2>/dev/null || true

docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 \
  -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w "$REMOTE_REPO" \
  "$IMAGE" \
  python3 scripts/spark_stage6_p2o_packed_prefill_rc_smoke.py \
    --model "$MODEL" \
    --preset "$PRESET" \
    --max-new "$MAX_NEW" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --json-out "$REMOTE_RUN_DIR/result.json" \
  > "$REMOTE_RUN_DIR/run.log" 2>&1

nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader > "$REMOTE_RUN_DIR/nvidia_smi_after.txt" 2>/dev/null || true
REMOTE

mkdir -p "$LOCAL_RUN_DIR"
rsync -av "${HOST}:${REMOTE_RUN_DIR}/" "$LOCAL_RUN_DIR/"

SUMMARY_ARGS=("$LOCAL_RUN_DIR/result.json" "--markdown-out" "$LOCAL_RUN_DIR/summary.md")
if [[ "$STRICT" == "1" ]]; then
  SUMMARY_ARGS+=("--strict-exit")
fi
python3 scripts/summarize_stage6_p2o_rc_smoke.py "${SUMMARY_ARGS[@]}"

echo "[p2o] local_run_dir=${LOCAL_RUN_DIR}"
