#!/usr/bin/env bash
set -Eeuo pipefail

# Thinking-off quality refresh for Qwen3.6-35B-A3B APEX-MTP I-Balanced.
# Run on dgx-spark. This script intentionally does not pass enable_thinking and
# uses the existing short-answer OpenAI-compatible MMLU/GPQA evaluators.

BASE_URL="${BASE_URL:-http://127.0.0.1:18098/v1}"
MODEL="${MODEL:-Qwen3.6-35B-A3B-APEX-I-Balanced.gguf}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/home/merkyor/eval/reports/apex_thinkoff_${STAMP}}"

EVAL_ROOT="${EVAL_ROOT:-/home/merkyor/eval}"
LYNN_SCRIPTS="${LYNN_SCRIPTS:-${EVAL_ROOT}/lynn-engine/scripts}"
MMLU_CSV_DIR="${MMLU_CSV_DIR:-/home/merkyor/lynn-nemotron-eval/mmlu_csv}"
GPQA_CSV="${GPQA_CSV:-/home/merkyor/quality-eval-20260517/datasets/gpqa/gpqa_diamond.csv}"

mkdir -p "$OUT_ROOT"
echo "$OUT_ROOT" > /home/merkyor/eval/reports/apex_thinkoff_latest_run_dir.txt

exec > >(tee -a "${OUT_ROOT}/run.log") 2>&1

echo "[apex-thinkoff] start $(date -Is)"
echo "[apex-thinkoff] base_url=${BASE_URL}"
echo "[apex-thinkoff] model=${MODEL}"
echo "[apex-thinkoff] out=${OUT_ROOT}"
echo "[apex-thinkoff] service=$(systemctl is-active lynn-apex-mtp-llamacpp.service || true)"
curl -fsS --max-time 10 "${BASE_URL%/v1}/health" || true
echo
ps -eo pid,etime,%mem,rss,cmd | grep llama-server | grep -v grep || true

echo
echo "========== mmlu500_5shot_thinkoff =========="
python3 "${LYNN_SCRIPTS}/openai_mmlu_500_5shot_eval.py" \
  --data-dir "$MMLU_CSV_DIR" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --out "${OUT_ROOT}/mmlu500_5shot_thinkoff.jsonl" \
  --sample 500 \
  --seed 20260519 \
  --shots 5 \
  --concurrency 4 \
  --timeout 120

echo
echo "========== gpqa198_thinkoff =========="
python3 "${LYNN_SCRIPTS}/openai_gpqa_diamond_eval.py" \
  --csv "$GPQA_CSV" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --out "${OUT_ROOT}/gpqa198_thinkoff.jsonl" \
  --concurrency 2 \
  --timeout 120

echo
echo "========== summaries =========="
find "$OUT_ROOT" -maxdepth 1 -name "*.summary.json" -print -exec cat {} \;
echo "[apex-thinkoff] end $(date -Is)"
curl -fsS --max-time 10 "${BASE_URL%/v1}/health" || true
echo
systemctl is-active lynn-apex-mtp-llamacpp.service || true
