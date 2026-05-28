#!/usr/bin/env bash
set -Eeuo pipefail

# Run on dgx-spark. This intentionally serializes the long 32K thinking-on
# quality gates so the production fallback service does not get surprised by
# multiple huge KV-cache jobs at once.

BASE_URL="${BASE_URL:-http://127.0.0.1:18098/v1}"
MODEL="${MODEL:-Qwen3.6-35B-A3B-APEX-I-Balanced.gguf}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/home/merkyor/eval/reports/apex_quality32k_${STAMP}}"

EVAL_ROOT="${EVAL_ROOT:-/home/merkyor/eval}"
LYNN_SCRIPTS="${LYNN_SCRIPTS:-${EVAL_ROOT}/lynn-engine/scripts}"
V9_HARNESS="${V9_HARNESS:-${EVAL_ROOT}/lynn-llm-benchmarks/v9/scripts/harness_v9.py}"
PROMPTS_DIR="${PROMPTS_DIR:-${EVAL_ROOT}/eval_prompts}"
MMLU_CSV_DIR="${MMLU_CSV_DIR:-/home/merkyor/lynn-nemotron-eval/mmlu_csv}"
GPQA_CSV="${GPQA_CSV:-/home/merkyor/quality-eval-20260517/datasets/gpqa/gpqa_diamond.csv}"

mkdir -p "$OUT_ROOT"
echo "$OUT_ROOT" > /home/merkyor/eval/reports/apex_quality32k_latest_run_dir.txt

exec > >(tee -a "${OUT_ROOT}/run.log") 2>&1

echo "[apex-quality32k] start $(date -Is)"
echo "[apex-quality32k] base_url=${BASE_URL}"
echo "[apex-quality32k] model=${MODEL}"
echo "[apex-quality32k] out=${OUT_ROOT}"

curl -fsS --max-time 10 "${BASE_URL%/v1}/health" || true
python3 - <<'PY'
import shutil
for mod in ["requests", "pyarrow"]:
    print(f"[apex-quality32k] python module {mod}: checking")
    __import__(mod)
print("[apex-quality32k] python deps ok")
PY

echo "[apex-quality32k] service process"
ps -eo pid,etime,%mem,rss,cmd | grep llama-server | grep -v grep || true
free -h || true
df -h /home || true

run_step() {
  local name="$1"
  shift
  echo
  echo "========== ${name} =========="
  echo "[apex-quality32k] ${name} start $(date -Is)"
  "$@"
  echo "[apex-quality32k] ${name} done $(date -Is)"
}

run_step "smoke_gpqa2_parse_4k" \
  python3 "${LYNN_SCRIPTS}/openai_mcq_thinking32_eval.py" \
    --task gpqa \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --gpqa-csv "$GPQA_CSV" \
    --out "${OUT_ROOT}/smoke_gpqa2_parse_4k.jsonl" \
    --limit 2 \
    --max-tokens 4096 \
    --concurrency 1 \
    --timeout 900

run_step "toolcall_v8_stage1_thinking_on_32k" \
  python3 "${EVAL_ROOT}/scripts/toolcall_runner.py" \
    --data "${PROMPTS_DIR}/stage1_tool_calling.jsonl" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --out "${OUT_ROOT}/toolcall_v8_stage1_thinking_on_32k.jsonl" \
    --enable-thinking \
    --max-tokens 32768 \
    --timeout 3600

run_step "toolcall_v8_stage5_coding_thinking_on_32k" \
  python3 "${EVAL_ROOT}/scripts/toolcall_runner.py" \
    --data "${PROMPTS_DIR}/stage5_coding.jsonl" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --out "${OUT_ROOT}/toolcall_v8_stage5_coding_thinking_on_32k.jsonl" \
    --enable-thinking \
    --max-tokens 32768 \
    --timeout 3600

run_step "v8_stage4_research_thinking_on_32k" \
  python3 - "$BASE_URL" "$MODEL" "${PROMPTS_DIR}/stage4_research.jsonl" "${OUT_ROOT}/v8_stage4_research_thinking_on_32k.jsonl" <<'PY'
import json
import sys
import time
import urllib.request

base_url, model, data_path, out_path = sys.argv[1:5]
rows = [json.loads(line) for line in open(data_path, encoding="utf-8") if line.strip()]

def call(user: str, max_tokens: int = 32768, timeout: int = 3600) -> tuple[str, dict, float, str | None]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        msg = data.get("choices", [{}])[0].get("message", {})
        text = ((msg.get("reasoning_content") or msg.get("reasoning") or "") + "\n\n" + (msg.get("content") or "")).strip()
        return text, data.get("usage", {}) or {}, time.time() - t0, None
    except Exception as exc:
        return "", {}, time.time() - t0, repr(exc)

results = []
with open(out_path, "w", encoding="utf-8") as f:
    for item in rows:
        text, usage, elapsed, error = call(item["user"])
        expected = item.get("expected") or {}
        min_chars = int(expected.get("min_chars") or 0)
        must_have = list(expected.get("must_have_keys") or [])
        post = text.split("</think>", 1)[-1] if "</think>" in text else text
        ok = bool(text) and len(post) >= min_chars and all(key in post for key in must_have) and not error
        row = {
            "id": item.get("id"),
            "category": item.get("category"),
            "ok": ok,
            "post_reasoning_chars": len(post),
            "raw_chars": len(text),
            "must_have_missing": [key for key in must_have if key not in post],
            "usage": usage,
            "elapsed_sec": round(elapsed, 3),
            "error": error,
            "raw_head": text[:512],
            "raw_tail": text[-512:] if text else "",
        }
        results.append(row)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        print(f"[v8-stage4] {row['id']} ok={ok} chars={row['post_reasoning_chars']} elapsed={row['elapsed_sec']}", flush=True)

summary = {
    "model": model,
    "endpoint": base_url,
    "n": len(results),
    "passed": sum(1 for row in results if row["ok"]),
    "pass_rate": sum(1 for row in results if row["ok"]) / len(results) if results else 0,
    "elapsed_sec": round(sum(row["elapsed_sec"] for row in results), 3),
}
summary_path = out_path.replace(".jsonl", ".summary.json")
open(summary_path, "w", encoding="utf-8").write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

run_step "v9_all_runs2_thinking_on_32k" \
  python3 - "$V9_HARNESS" "${OUT_ROOT}/v9_all_runs2_thinking_on_32k.json" <<'PY'
import importlib.util
import sys

harness_path, out_path = sys.argv[1:3]
spec = importlib.util.spec_from_file_location("harness_v9_32k", harness_path)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
for provider in mod.PROVIDERS:
    if provider.get("name") == "Qwen3.6-A3B-APEX-I-Balanced (Spark)":
        provider["max_tokens"] = 32768
        provider["extra"] = {"chat_template_kwargs": {"enable_thinking": True}, "enable_thinking": True}
        provider["url"] = "http://127.0.0.1:18098/v1/chat/completions"
        provider["model"] = "Qwen3.6-35B-A3B-APEX-I-Balanced.gguf"
        break
sys.argv = [
    harness_path,
    "--provider", "Qwen3.6-A3B-APEX-I-Balanced (Spark)",
    "--all",
    "--runs", "2",
    "--timeout", "3600",
    "--out", out_path,
]
raise SystemExit(mod.main())
PY

run_step "mmlu500_5shot_thinking_on_32k" \
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

run_step "gpqa198_thinking_on_32k" \
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
echo "========== summaries =========="
find "$OUT_ROOT" -maxdepth 1 -name "*.summary.json" -print -exec cat {} \;
echo "[apex-quality32k] end $(date -Is)"
curl -fsS --max-time 10 "${BASE_URL%/v1}/health" || true
systemctl is-active lynn-apex-mtp-llamacpp.service || true
