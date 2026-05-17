#!/usr/bin/env bash
# Run Qwen3.6-35B-A3B Lynn-native W4A16 NVFP4 quality eval against an already
# loaded Spark evaluation container. This avoids restarting the 35B server when
# the generic runner's short smoke timeout exits after first-use kernel compile.

set -euo pipefail

CONT="${CONT:-lynn-eval-nvfp4-qwen36-w4a16-official-n5}"
PORT="${PORT:-18098}"
SERVED_NAME="${SERVED_NAME:-Qwen3.6-35B-A3B-W4A16-NVFP4-official-n5}"
CAND="${CAND:-nvfp4-qwen36-w4a16-official-n5}"
QE_ROOT="${QE_ROOT:-/home/merkyor/quality-eval-20260517}"

RESULTS_DIR="$QE_ROOT/results"
MMLU_OUT="$RESULTS_DIR/mmlu_${CAND}_n500.jsonl"
MMLU_SUMMARY="$RESULTS_DIR/mmlu_${CAND}_n500.summary.json"
GPQA_OUT="$RESULTS_DIR/gpqa_${CAND}.jsonl"
GPQA_SUMMARY="$RESULTS_DIR/gpqa_${CAND}.summary.json"

curl -s -m 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null

rm -f "$MMLU_OUT" "$MMLU_SUMMARY" "$GPQA_OUT" "$GPQA_SUMMARY"

sudo docker exec "$CONT" bash -lc "
set -euo pipefail
mkdir -p /tmp/datasets/mmlu /tmp/datasets/gpqa
cp /qe/scripts/mmlu_runner_v2.py /tmp/mmlu_runner_v2.py
cp /qe/scripts/gpqa_runner_v2.py /tmp/gpqa_runner_v2.py
cp /qe/datasets/mmlu/*.parquet /tmp/datasets/mmlu/ 2>/dev/null || true
cp /qe/datasets/gpqa/gpqa_diamond.csv /tmp/datasets/gpqa/gpqa_diamond.csv 2>/dev/null || true
python3 /tmp/mmlu_runner_v2.py \
  --data-dir /tmp/datasets/mmlu \
  --base-url http://127.0.0.1:${PORT}/v1 \
  --model ${SERVED_NAME} \
  --out /qe/results/mmlu_${CAND}_n500.jsonl \
  --concurrency 4 \
  --shots 5 \
  --sample 500
python3 /tmp/gpqa_runner_v2.py \
  --csv /tmp/datasets/gpqa/gpqa_diamond.csv \
  --base-url http://127.0.0.1:${PORT}/v1 \
  --model ${SERVED_NAME} \
  --out /qe/results/gpqa_${CAND}.jsonl \
  --concurrency 2
"

cat "$MMLU_SUMMARY"
cat "$GPQA_SUMMARY"
