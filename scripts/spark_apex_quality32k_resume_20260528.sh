#!/usr/bin/env bash
set -Eeuo pipefail

# Resume the 2026-05-28 APEX quality32k run after the first script hit a broken
# V9 harness copy. This starts at V9 and continues through MMLU500 + GPQA198.

BASE_URL="${BASE_URL:-http://127.0.0.1:18098/v1}"
MODEL="${MODEL:-Qwen3.6-35B-A3B-APEX-I-Balanced.gguf}"
OUT_ROOT="${OUT_ROOT:-/home/merkyor/eval/reports/apex_quality32k_20260528_121431}"
EVAL_ROOT="${EVAL_ROOT:-/home/merkyor/eval}"
LYNN_SCRIPTS="${LYNN_SCRIPTS:-${EVAL_ROOT}/lynn-engine/scripts}"
V9_HARNESS="${V9_HARNESS:-/home/merkyor/lynn-v9-bench/scripts/harness_v9.py}"
MMLU_CSV_DIR="${MMLU_CSV_DIR:-/home/merkyor/lynn-nemotron-eval/mmlu_csv}"
GPQA_CSV="${GPQA_CSV:-/home/merkyor/quality-eval-20260517/datasets/gpqa/gpqa_diamond.csv}"

mkdir -p "$OUT_ROOT"
exec >> "${OUT_ROOT}/run.log" 2>&1

echo
echo "[apex-quality32k-resume] start $(date -Is)"
echo "[apex-quality32k-resume] out=${OUT_ROOT}"
echo "[apex-quality32k-resume] service=$(systemctl is-active lynn-apex-mtp-llamacpp.service || true)"
curl -fsS --max-time 10 "${BASE_URL%/v1}/health" || true
echo

run_step() {
  local name="$1"
  shift
  echo
  echo "========== ${name} =========="
  echo "[apex-quality32k-resume] ${name} start $(date -Is)"
  "$@"
  echo "[apex-quality32k-resume] ${name} done $(date -Is)"
}

run_step "v9_all_runs2_thinking_on_32k_resume" \
  python3 - "$V9_HARNESS" "${OUT_ROOT}/v9_all_runs2_thinking_on_32k.json" <<'PY'
import importlib.util
import sys

harness_path, out_path = sys.argv[1:3]
spec = importlib.util.spec_from_file_location("harness_v9_32k_resume", harness_path)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
for provider in mod.PROVIDERS:
    if provider.get("name") == "Qwen3.6-A3B (Spark)":
        provider["max_tokens"] = 32768
        provider["url"] = "http://127.0.0.1:18098/v1/chat/completions"
        provider["model"] = "Qwen3.6-35B-A3B-APEX-I-Balanced.gguf"
        provider["extra"] = {"chat_template_kwargs": {"enable_thinking": True}, "enable_thinking": True}
        break
sys.argv = [
    harness_path,
    "--provider", "Qwen3.6-A3B (Spark)",
    "--all",
    "--runs", "2",
    "--timeout", "3600",
    "--out", out_path,
]
raise SystemExit(mod.main())
PY

run_step "mmlu500_5shot_thinking_on_32k_resume" \
  python3 "${LYNN_SCRIPTS}/openai_mmlu_csv_thinking32_eval.py" \
    --data-dir "$MMLU_CSV_DIR" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --out "${OUT_ROOT}/mmlu500_5shot_thinking_on_32k.jsonl" \
    --sample 500 \
    --seed 20260519 \
    --shots 5 \
    --max-tokens 32768 \
    --timeout 3600

run_step "gpqa198_thinking_on_32k_resume" \
  python3 "${LYNN_SCRIPTS}/openai_mcq_thinking32_eval.py" \
    --task gpqa \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --gpqa-csv "$GPQA_CSV" \
    --out "${OUT_ROOT}/gpqa198_thinking_on_32k.jsonl" \
    --max-tokens 32768 \
    --concurrency 1 \
    --timeout 3600

echo
echo "========== resume summaries =========="
find "$OUT_ROOT" -maxdepth 1 -name "*.summary.json" -print -exec cat {} \;
echo "[apex-quality32k-resume] end $(date -Is)"
curl -fsS --max-time 10 "${BASE_URL%/v1}/health" || true
echo
systemctl is-active lynn-apex-mtp-llamacpp.service || true
