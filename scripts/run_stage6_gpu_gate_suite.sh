#!/usr/bin/env bash
# Run the current Stage 6 GPU gates as one headless suite.
#
# This is the orchestration layer: it does not implement benchmarks itself.
# It calls the stricter per-gate wrappers, groups their artifacts under one
# suite directory, and keeps running so failures still leave evidence.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_stage6_gpu_gate_suite.sh [options]

Options:
  --host HOST                  SSH host alias. Default: $LYNN_SPARK_HOST or dgx-spark.
  --model PATH                 Model path on Spark.
  --image IMAGE                Docker image. Default: lynn-eval-base:cu13.
  --remote-repo PATH           Spark repo path. Default: /home/merkyor/lynn-engine.
  --local-root PATH            Local artifact root. Default: reports/stage6.
  --expect-head COMMIT         Expected Spark repo HEAD. Default: local HEAD.
  --allow-provenance-mismatch  Pass through to child gates.
  --no-strict                  Keep running and exit 0 even if a child gate fails.
  --dry-run                    Print child commands without executing them.
  --skip-p2o-basic             Skip P2-O basic preset.
  --skip-p2o-rc-mini           Skip P2-O rc-mini preset.
  --skip-p3a                   Skip P3-A grouped-MoE contract probe.
  --skip-p3b                   Skip P3-B selected-prefill composition gate.
  --skip-p3c                   Skip P3-C resident-prompt gate.
  --p2o-max-new N              P2-O generated tokens. Default: 8.
  --p2o-max-seq-len N          P2-O max_seq_len. Default: 2048.
  --p3a-layer N                P3-A layer. Default: 0.
  --p3a-batches CSV            P3-A batches. Default: 1,16,64.
  --p3b-layers SPEC            P3-B layer spec. Default: 0-3.
  --p3b-tokens CSV             P3-B sequence lengths. Default: 16,64.
  --p3c-preset basic|rc-mini   P3-C prompt preset. Default: basic.
  -h, --help                   Show this help.

Environment overrides:
  LYNN_SPARK_HOST
  LYNN_STAGE6_MODEL
  LYNN_SPARK_IMAGE
  LYNN_SPARK_REPO
  LYNN_STAGE6_LOCAL_OUT
  LYNN_STAGE6_EXPECT_HEAD
USAGE
}

shell_join() {
  local out="" arg
  for arg in "$@"; do
    if [[ -n "$out" ]]; then
      out+=" "
    fi
    out+="$(printf '%q' "$arg")"
  done
  printf '%s' "$out"
}

HOST="${LYNN_SPARK_HOST:-dgx-spark}"
MODEL="${LYNN_STAGE6_MODEL:-/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526}"
IMAGE="${LYNN_SPARK_IMAGE:-lynn-eval-base:cu13}"
REMOTE_REPO="${LYNN_SPARK_REPO:-/home/merkyor/lynn-engine}"
LOCAL_ROOT="${LYNN_STAGE6_LOCAL_OUT:-reports/stage6}"
EXPECTED_HEAD="${LYNN_STAGE6_EXPECT_HEAD:-$(git rev-parse HEAD 2>/dev/null || true)}"
STRICT="1"
DRY_RUN="0"
REQUIRE_PROVENANCE="1"
RUN_P2O_BASIC="1"
RUN_P2O_RC_MINI="1"
RUN_P3A="1"
RUN_P3B="1"
RUN_P3C="1"
P2O_MAX_NEW="8"
P2O_MAX_SEQ_LEN="2048"
P3A_LAYER="0"
P3A_BATCHES="1,16,64"
P3B_LAYERS="0-3"
P3B_TOKENS="16,64"
P3C_PRESET="basic"

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
    --allow-provenance-mismatch)
      REQUIRE_PROVENANCE="0"
      shift
      ;;
    --no-strict)
      STRICT="0"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    --skip-p2o-basic)
      RUN_P2O_BASIC="0"
      shift
      ;;
    --skip-p2o-rc-mini)
      RUN_P2O_RC_MINI="0"
      shift
      ;;
    --skip-p3a)
      RUN_P3A="0"
      shift
      ;;
    --skip-p3b)
      RUN_P3B="0"
      shift
      ;;
    --skip-p3c)
      RUN_P3C="0"
      shift
      ;;
    --p2o-max-new)
      P2O_MAX_NEW="$2"
      shift 2
      ;;
    --p2o-max-seq-len)
      P2O_MAX_SEQ_LEN="$2"
      shift 2
      ;;
    --p3a-layer)
      P3A_LAYER="$2"
      shift 2
      ;;
    --p3a-batches)
      P3A_BATCHES="$2"
      shift 2
      ;;
    --p3b-layers)
      P3B_LAYERS="$2"
      shift 2
      ;;
    --p3b-tokens|--p3b-seq-lens)
      P3B_TOKENS="$2"
      shift 2
      ;;
    --p3c-preset)
      P3C_PRESET="$2"
      shift 2
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
RUN_NAME="stage6_gpu_gate_suite_${STAMP}_$$"
SUITE_DIR="${LOCAL_ROOT}/${RUN_NAME}"
STATUS_TSV="${SUITE_DIR}/suite_status.tsv"
SUMMARY_MD="${SUITE_DIR}/summary.md"
COMMANDS_SH="${SUITE_DIR}/commands.sh"

mkdir -p "$SUITE_DIR"
{
  echo "local_head=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "expected_head=${EXPECTED_HEAD}"
  echo "host=${HOST}"
  echo "model=${MODEL}"
  echo "image=${IMAGE}"
  echo "remote_repo=${REMOTE_REPO}"
  echo "strict=${STRICT}"
  echo "dry_run=${DRY_RUN}"
  echo "created_at=${STAMP}"
} > "${SUITE_DIR}/suite_meta.env"
git status --short > "${SUITE_DIR}/local_git_status.txt" 2>/dev/null || true
printf 'step\tstatus\texit_code\n' > "$STATUS_TSV"
: > "$COMMANDS_SH"

common_args=(
  --host "$HOST"
  --model "$MODEL"
  --image "$IMAGE"
  --remote-repo "$REMOTE_REPO"
  --local-root "$SUITE_DIR"
  --expect-head "$EXPECTED_HEAD"
)
if [[ "$REQUIRE_PROVENANCE" == "0" ]]; then
  common_args+=(--allow-provenance-mismatch)
fi
# Do not pass --no-strict to child gates. Suite --no-strict only controls the
# suite's final exit code; child wrappers must stay strict so PASS/FAIL remains
# valid evidence for predecessor-gated steps such as P3-B.

run_step() {
  local name="$1"
  shift
  local cmd=("$@")
  local rendered
  rendered="$(shell_join "${cmd[@]}")"
  echo "$rendered" >> "$COMMANDS_SH"
  echo "[suite] ${name}: ${rendered}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s\t%s\t%s\n' "$name" "DRY_RUN" "0" >> "$STATUS_TSV"
    return 0
  fi
  set +e
  "${cmd[@]}"
  local code=$?
  set -e
  if [[ "$code" == "0" ]]; then
    printf '%s\t%s\t%s\n' "$name" "PASS" "$code" >> "$STATUS_TSV"
  else
    printf '%s\t%s\t%s\n' "$name" "FAIL" "$code" >> "$STATUS_TSV"
  fi
  return "$code"
}

failures=0
P2O_BASIC_STATUS="SKIP"
P2O_RC_MINI_STATUS="SKIP"
P3A_STATUS="SKIP"
P3B_STATUS="SKIP"
if [[ "$RUN_P2O_BASIC" == "1" ]]; then
  run_step "p2o-basic" \
    scripts/run_spark_stage6_p2o_rc_smoke.sh \
      "${common_args[@]}" \
      --preset basic \
      --max-new "$P2O_MAX_NEW" \
      --max-seq-len "$P2O_MAX_SEQ_LEN" && P2O_BASIC_STATUS="PASS" || {
        P2O_BASIC_STATUS="FAIL"
        failures=$((failures + 1))
      }
else
  printf '%s\t%s\t%s\n' "p2o-basic" "SKIP" "0" >> "$STATUS_TSV"
fi

if [[ "$RUN_P2O_RC_MINI" == "1" ]]; then
  run_step "p2o-rc-mini" \
    scripts/run_spark_stage6_p2o_rc_smoke.sh \
      "${common_args[@]}" \
      --preset rc-mini \
      --max-new "$P2O_MAX_NEW" \
      --max-seq-len "$P2O_MAX_SEQ_LEN" && P2O_RC_MINI_STATUS="PASS" || {
        P2O_RC_MINI_STATUS="FAIL"
        failures=$((failures + 1))
      }
else
  printf '%s\t%s\t%s\n' "p2o-rc-mini" "SKIP" "0" >> "$STATUS_TSV"
fi

if [[ "$RUN_P3A" == "1" ]]; then
  run_step "p3a-contract" \
    scripts/run_spark_stage6_p3a_contract_probe.sh \
      "${common_args[@]}" \
      --layer "$P3A_LAYER" \
      --batches "$P3A_BATCHES" && P3A_STATUS="PASS" || {
        P3A_STATUS="FAIL"
        failures=$((failures + 1))
      }
else
  printf '%s\t%s\t%s\n' "p3a-contract" "SKIP" "0" >> "$STATUS_TSV"
fi

if [[ "$RUN_P3B" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" || ( "$STRICT" == "1" && "$P2O_BASIC_STATUS" == "PASS" && "$P2O_RC_MINI_STATUS" == "PASS" && "$P3A_STATUS" == "PASS" ) ]]; then
    run_step "p3b-selected-prefill" \
      scripts/run_spark_stage6_p3b_selected_prefill_gate.sh \
        "${common_args[@]}" \
        --layers "$P3B_LAYERS" \
        --tokens "$P3B_TOKENS" \
        --predecessors-pass && P3B_STATUS="PASS" || {
      P3B_STATUS="FAIL"
      failures=$((failures + 1))
    }
  else
    echo "[suite] p3b-selected-prefill: skipped because strict predecessor PASS evidence is unavailable"
    printf '%s\t%s\t%s\n' "p3b-selected-prefill" "SKIP" "0" >> "$STATUS_TSV"
  fi
else
  printf '%s\t%s\t%s\n' "p3b-selected-prefill" "SKIP" "0" >> "$STATUS_TSV"
fi

if [[ "$RUN_P3C" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" || ( "$STRICT" == "1" && "$P3B_STATUS" == "PASS" ) ]]; then
    run_step "p3c-resident-prompt" \
      scripts/run_spark_stage6_p3c_resident_prompt_gate.sh \
        "${common_args[@]}" \
        --preset "$P3C_PRESET" \
        --max-new "$P2O_MAX_NEW" \
        --max-seq-len "$P2O_MAX_SEQ_LEN" \
        --p3b-pass && true || failures=$((failures + 1))
  else
    echo "[suite] p3c-resident-prompt: skipped because strict P3-B PASS evidence is unavailable"
    printf '%s\t%s\t%s\n' "p3c-resident-prompt" "SKIP" "0" >> "$STATUS_TSV"
  fi
else
  printf '%s\t%s\t%s\n' "p3c-resident-prompt" "SKIP" "0" >> "$STATUS_TSV"
fi

{
  echo "# Stage 6 GPU Gate Suite"
  echo
  echo "| Field | Value |"
  echo "|---|---|"
  echo "| Host | \`${HOST}\` |"
  echo "| Model | \`${MODEL}\` |"
  echo "| Expected HEAD | \`${EXPECTED_HEAD:-none}\` |"
  echo "| Strict | \`${STRICT}\` |"
  echo "| Dry run | \`${DRY_RUN}\` |"
  echo "| Failures | \`${failures}\` |"
  echo
  echo "## Steps"
  echo
  echo "| Step | Status | Exit code |"
  echo "|---|---|---:|"
  tail -n +2 "$STATUS_TSV" | while IFS=$'\t' read -r step status code; do
    echo "| ${step} | ${status} | ${code} |"
  done
  echo
  echo "## Commands"
  echo
  echo '```bash'
  cat "$COMMANDS_SH"
  echo '```'
} > "$SUMMARY_MD"

if [[ -f scripts/write_stage6_gpu_gate_suite_report.py ]]; then
  python3 scripts/write_stage6_gpu_gate_suite_report.py \
    "$SUITE_DIR" \
    --report-out "$SUITE_DIR/report.md" || true
fi

echo "[suite] artifacts: ${SUITE_DIR}"
if [[ "$STRICT" == "1" && "$failures" -ne 0 ]]; then
  exit 2
fi
exit 0
