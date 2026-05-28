#!/usr/bin/env bash
set -Eeuo pipefail

# Quality gates for the self-quantized Qwen3.6-35B-A3B Q4_K_M imatrix GGUF
# with embedded APEX-MTP. Run on dgx-spark after the artifact is served on
# 18099. The production 18098 Brain V2 fallback is only health-checked here.

BASE_URL="${BASE_URL:-http://127.0.0.1:18099/v1}"
MODEL="${MODEL:-qwen36-35b-a3b-apex-mtp-q4km}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/home/merkyor/eval/reports/qwen36_q4km_mtp_quality_${STAMP}}"

EVAL_ROOT="${EVAL_ROOT:-/home/merkyor/eval}"
LYNN_SCRIPTS="${LYNN_SCRIPTS:-${EVAL_ROOT}/lynn-engine/scripts}"
PROMPTS_DIR="${PROMPTS_DIR:-${EVAL_ROOT}/eval_prompts}"
MMLU_CSV_DIR="${MMLU_CSV_DIR:-/home/merkyor/lynn-nemotron-eval/mmlu_csv}"
GPQA_CSV="${GPQA_CSV:-/home/merkyor/quality-eval-20260517/datasets/gpqa/gpqa_diamond.csv}"

MMLU_SAMPLE="${MMLU_SAMPLE:-500}"
MMLU_SEED="${MMLU_SEED:-20260519}"
MMLU_SHOTS="${MMLU_SHOTS:-5}"
THINKING_MAX_TOKENS="${THINKING_MAX_TOKENS:-32768}"
THINKOFF_CONCURRENCY="${THINKOFF_CONCURRENCY:-1}"
THINKON_CONCURRENCY="${THINKON_CONCURRENCY:-1}"
RUN_TOOLCALL="${RUN_TOOLCALL:-1}"

mkdir -p "$OUT_ROOT"
echo "$OUT_ROOT" > /home/merkyor/eval/reports/qwen36_q4km_mtp_quality_latest_run_dir.txt

exec > >(tee -a "${OUT_ROOT}/run.log") 2>&1

echo "[q4km-mtp-quality] start $(date -Is)"
echo "[q4km-mtp-quality] base_url=${BASE_URL}"
echo "[q4km-mtp-quality] model=${MODEL}"
echo "[q4km-mtp-quality] out=${OUT_ROOT}"
echo "[q4km-mtp-quality] production_fallback=$(systemctl is-active lynn-apex-mtp-llamacpp.service || true)"
curl -fsS --max-time 10 http://127.0.0.1:18098/health || true
echo
curl -fsS --max-time 10 "${BASE_URL%/v1}/health"
echo
ps -eo pid,etime,%mem,rss,cmd | grep llama-server | grep -v grep || true
free -h || true

run_step() {
  local name="$1"
  shift
  echo
  echo "========== ${name} =========="
  echo "[q4km-mtp-quality] ${name} start $(date -Is)"
  "$@"
  echo "[q4km-mtp-quality] ${name} done $(date -Is)"
}

run_step "mmlu500_5shot_thinkoff" \
  python3 "${LYNN_SCRIPTS}/openai_mmlu_500_5shot_eval.py" \
    --data-dir "$MMLU_CSV_DIR" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --out "${OUT_ROOT}/mmlu500_5shot_thinkoff.jsonl" \
    --sample "$MMLU_SAMPLE" \
    --seed "$MMLU_SEED" \
    --shots "$MMLU_SHOTS" \
    --concurrency "$THINKOFF_CONCURRENCY" \
    --timeout 180 \
    --disable-thinking \
    --append-no-think

run_step "gpqa198_thinkoff" \
  python3 "${LYNN_SCRIPTS}/openai_gpqa_diamond_eval.py" \
    --csv "$GPQA_CSV" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --out "${OUT_ROOT}/gpqa198_thinkoff.jsonl" \
    --concurrency "$THINKOFF_CONCURRENCY" \
    --timeout 180 \
    --disable-thinking \
    --append-no-think

run_step "mmlu500_5shot_thinking_on_32k" \
  python3 "${LYNN_SCRIPTS}/openai_mmlu_csv_thinking32_eval.py" \
    --data-dir "$MMLU_CSV_DIR" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --out "${OUT_ROOT}/mmlu500_5shot_thinking_on_32k.jsonl" \
    --sample "$MMLU_SAMPLE" \
    --seed "$MMLU_SEED" \
    --shots "$MMLU_SHOTS" \
    --max-tokens "$THINKING_MAX_TOKENS" \
    --timeout 3600

run_step "gpqa198_thinking_on_32k" \
  python3 "${LYNN_SCRIPTS}/openai_mcq_thinking32_eval.py" \
    --task gpqa \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --gpqa-csv "$GPQA_CSV" \
    --out "${OUT_ROOT}/gpqa198_thinking_on_32k.jsonl" \
    --max-tokens "$THINKING_MAX_TOKENS" \
    --concurrency "$THINKON_CONCURRENCY" \
    --timeout 3600

if [ "$RUN_TOOLCALL" = "1" ]; then
  run_step "toolcall_v8_stage1_thinking_on_32k" \
    python3 "${EVAL_ROOT}/scripts/toolcall_runner.py" \
      --data "${PROMPTS_DIR}/stage1_tool_calling.jsonl" \
      --base-url "$BASE_URL" \
      --model "$MODEL" \
      --out "${OUT_ROOT}/toolcall_v8_stage1_thinking_on_32k.jsonl" \
      --enable-thinking \
      --max-tokens "$THINKING_MAX_TOKENS" \
      --timeout 3600

  run_step "toolcall_v8_stage5_coding_thinking_on_32k" \
    python3 "${EVAL_ROOT}/scripts/toolcall_runner.py" \
      --data "${PROMPTS_DIR}/stage5_coding.jsonl" \
      --base-url "$BASE_URL" \
      --model "$MODEL" \
      --out "${OUT_ROOT}/toolcall_v8_stage5_coding_thinking_on_32k.jsonl" \
      --enable-thinking \
      --max-tokens "$THINKING_MAX_TOKENS" \
      --timeout 3600
fi

echo
echo "========== summaries =========="
find "$OUT_ROOT" -maxdepth 1 -name "*.summary.json" -print -exec cat {} \;
echo "[q4km-mtp-quality] end $(date -Is)"
curl -fsS --max-time 10 "${BASE_URL%/v1}/health" || true
echo
echo "[q4km-mtp-quality] production_fallback=$(systemctl is-active lynn-apex-mtp-llamacpp.service || true)"
curl -fsS --max-time 10 http://127.0.0.1:18098/health || true
echo
