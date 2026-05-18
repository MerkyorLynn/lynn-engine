#!/usr/bin/env bash
set -euo pipefail

# R6000 llama.cpp baseline for Qwen3.5-9B Q4_K_M GGUF.
# Measures:
#   1. single-stream decode TPS (128/256/512)
#   2. concurrent total TPS (2/4/8)
#   3. long-context prefill+decode smoke (4k/16k/32k)
#
# GGUF auto-discovery:
#   1) --gguf / $GGUF
#   2) $MODEL_ROOT/Qwen3.5-9B-Q4_K_M.gguf
#   3) $MODEL_ROOT/*Qwen3.5*9B*Q4*K*M*.gguf (glob)
#
# If GGUF not found:
#   Writes PENDING_DOWNLOAD report and prints download command templates.
#
# Download entry points (set one):
#   HF_REPO  — HuggingFace repo ID (e.g. Qwen/Qwen3.5-9B-GGUF)
#   MS_REPO  — ModelScope repo ID
#   GGUF_URL — direct download URL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/models}"
REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18197}"
SERVED_NAME="${SERVED_NAME:-Qwen3.5-9B-Q4KM}"
CTX_SIZE="${CTX_SIZE:-32768}"
THREADS="${THREADS:-12}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
PARALLEL="${PARALLEL:-8}"
SINGLE_MAX_TOKENS="${SINGLE_MAX_TOKENS:-"128 256 512"}"
CONCURRENCY="${CONCURRENCY:-"2 4 8"}"
CONCURRENT_MAX_TOKENS="${CONCURRENT_MAX_TOKENS:-256}"
LONG_CONTEXT_CHARS="${LONG_CONTEXT_CHARS:-"4096 16384 32768"}"
LONG_CONTEXT_MAX_TOKENS="${LONG_CONTEXT_MAX_TOKENS:-128}"
LLAMA_SERVER="${LLAMA_SERVER:-}"
LLAMA_BENCH="${LLAMA_BENCH:-}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"
REASONING="${REASONING:-auto}"
QUALITY="${QUALITY:-0}"
MMLU_DATA_DIR="${MMLU_DATA_DIR:-/tmp/datasets/mmlu}"
GPQA_CSV="${GPQA_CSV:-/tmp/datasets/gpqa/gpqa_diamond.csv}"
HF_REPO="${HF_REPO:-}"
MS_REPO="${MS_REPO:-}"
GGUF_URL="${GGUF_URL:-}"
GGUF="${GGUF:-}"

mkdir -p "$REPORT_ROOT"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || command -v python)"
fi

# ---------------------------------------------------------------------------
# GGUF discovery
# ---------------------------------------------------------------------------
GGUF_FOUND=""
if [[ -n "$GGUF" && -s "$GGUF" ]]; then
  GGUF_FOUND="$GGUF"
else
  # Try standard name
  local_std="$MODEL_ROOT/Qwen3.5-9B-Q4_K_M.gguf"
  if [[ -s "$local_std" ]]; then
    GGUF_FOUND="$local_std"
  else
    # Glob search
    for candidate in "$MODEL_ROOT"/*Qwen3.5*9B*Q4*K*M*.gguf \
                     "$MODEL_ROOT"/*qwen3.5*9b*q4*k*m*.gguf; do
      if [[ -s "${candidate:-}" ]]; then
        GGUF_FOUND="$candidate"
        break
      fi
    done
    if [[ -z "$GGUF_FOUND" ]]; then
      for candidate in "$MODEL_ROOT"/Qwen3.5-9B-GGUF/*.gguf \
                       "$MODEL_ROOT"/Qwen3.5*9B*GGUF/*.gguf; do
        if [[ -s "${candidate:-}" ]]; then
          GGUF_FOUND="$candidate"
          break
        fi
      done
    fi
  fi
fi

OUT="$REPORT_ROOT/r6000_qwen35_9b_q4km_baseline_${STAMP}.json"

# ---------------------------------------------------------------------------
# Pending download handler
# ---------------------------------------------------------------------------
if [[ -z "$GGUF_FOUND" ]]; then
  echo "[q4km-9b] GGUF not found in $MODEL_ROOT" >&2
  echo "[q4km-9b] Writing PENDING_DOWNLOAD report to $OUT" >&2

  cat > "$OUT" <<ENDJSON
{
  "schema": "lynn-qwen35-9b-q4km-baseline-v1",
  "created": "$(date -Iseconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z)",
  "status": "PENDING_DOWNLOAD",
  "model_id": "Qwen3.5-9B",
  "quant": "Q4_K_M",
  "model_path": null,
  "size_gib": null,
  "llama_cpp_binary": null,
  "git_rev": null,
  "single_tps": {},
  "concurrent_tps": {},
  "long_context": {},
  "errors": ["GGUF not found. Set GGUF, HF_REPO, MS_REPO, or GGUF_URL."],
  "recommendation": {
    "hf_repo": "Qwen/Qwen3.5-9B-GGUF",
    "ms_repo": "Qwen/Qwen3.5-9B-GGUF",
    "filename": "qwen3.5-9b-q4_k_m.gguf",
    "download_commands": [
      "# HuggingFace CLI:",
      "huggingface-cli download Qwen/Qwen3.5-9B-GGUF qwen3.5-9b-q4_k_m.gguf --local-dir $MODEL_ROOT",
      "",
      "# ModelScope CLI:",
      "modelscope download Qwen/Qwen3.5-9B-GGUF qwen3.5-9b-q4_k_m.gguf --local_dir $MODEL_ROOT",
      "",
      "# Direct URL:",
      "wget -O $MODEL_ROOT/Qwen3.5-9B-Q4_K_M.gguf '<GGUF_URL_HERE>'"
    ]
  }
}
ENDJSON

  # Also generate PENDING markdown
  OUT_MD="$REPORT_ROOT/r6000_qwen35_9b_q4km_baseline_${STAMP}.md"
  cat > "$OUT_MD" <<ENDMD
# Qwen3.5-9B Q4_K_M llama.cpp Baseline — PENDING_DOWNLOAD

**Status:** ⏳ PENDING_DOWNLOAD
**Created:** $(date -Iseconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z)

## Missing Asset

GGUF not found in \`$MODEL_ROOT\`.

### Recommended Download

\`\`\`bash
# HuggingFace (recommended)
huggingface-cli download Qwen/Qwen3.5-9B-GGUF qwen3.5-9b-q4_k_m.gguf \\
  --local-dir $MODEL_ROOT

# ModelScope (China mirror)
modelscope download Qwen/Qwen3.5-9B-GGUF qwen3.5-9b-q4_k_m.gguf \\
  --local_dir $MODEL_ROOT

# Or set GGUF_URL and re-run this script
export GGUF_URL="https://huggingface.co/Qwen/Qwen3.5-9B-GGUF/resolve/main/qwen3.5-9b-q4_k_m.gguf"
\`\`\`

## Pending Benchmarks

| Section | Status |
|---------|--------|
| Single TPS (128/256/512) | ⏳ PENDING |
| Concurrent TPS (2/4/8) | ⏳ PENDING |
| Long Context (4k/16k/32k) | ⏳ PENDING |

## Report

\`$OUT\`
ENDMD

  echo "[q4km-9b] PENDING reports written:" >&2
  echo "  JSON: $OUT" >&2
  echo "  MD:   $OUT_MD" >&2
  exit 0
fi

GGUF_SIZE_BYTES=$(stat -c%s "$GGUF_FOUND" 2>/dev/null || stat -f%z "$GGUF_FOUND" 2>/dev/null || echo 0)
GGUF_SIZE_GIB=$(echo "scale=2; $GGUF_SIZE_BYTES / 1073741824" | bc 2>/dev/null || echo "unknown")

echo "[q4km-9b] GGUF found: $GGUF_FOUND ($GGUF_SIZE_GIB GiB)"

# ---------------------------------------------------------------------------
# Auto-discover llama.cpp binaries
# ---------------------------------------------------------------------------
if [[ -z "$LLAMA_SERVER" ]]; then
  for candidate in \
    /root/autodl-tmp/llama.cpp/build-cuda/bin/llama-server \
    /root/autodl-tmp/llama.cpp/build/bin/llama-server \
    /root/autodl-tmp/llama.cpp/build/tools/server/llama-server; do
    if [[ -x "${candidate:-}" ]]; then
      LLAMA_SERVER="$candidate"
      break
    fi
  done
fi
if [[ -z "$LLAMA_SERVER" || ! -x "${LLAMA_SERVER:-}" ]]; then
  echo "[q4km-9b] llama-server binary not found" >&2
  echo "[q4km-9b] Build llama.cpp with: cmake -B build-cuda -DGGML_CUDA=ON && cmake --build build-cuda -j" >&2
  exit 3
fi

if [[ -z "$LLAMA_BENCH" ]]; then
  server_dir="$(dirname "$LLAMA_SERVER")"
  for candidate in \
    "${server_dir}/../llama-bench" \
    "${server_dir}/llama-bench" \
    /root/autodl-tmp/llama.cpp/build-cuda/bin/llama-bench \
    /root/autodl-tmp/llama.cpp/build/bin/llama-bench; do
    if [[ -x "${candidate:-}" ]]; then
      LLAMA_BENCH="$candidate"
      break
    fi
  done
fi

# Get git revision
LLAMA_GIT_REV=""
if [[ -d /root/autodl-tmp/llama.cpp/.git ]]; then
  LLAMA_GIT_REV=$(git -C /root/autodl-tmp/llama.cpp rev-parse --short HEAD 2>/dev/null || echo "unknown")
else
  LLAMA_GIT_REV="unknown"
fi

echo "[q4km-9b] server=$LLAMA_SERVER"
echo "[q4km-9b] bench=${LLAMA_BENCH:-not found}"
echo "[q4km-9b] llama.cpp rev=$LLAMA_GIT_REV"
echo "[q4km-9b] out=$OUT"

# ---------------------------------------------------------------------------
# Start server
# ---------------------------------------------------------------------------
SERVER_LOG="$REPORT_ROOT/r6000_qwen35_9b_q4km_server_${STAMP}.log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"$LLAMA_SERVER" \
  --model "$GGUF_FOUND" \
  --host "$HOST" \
  --port "$PORT" \
  --n-gpu-layers "$N_GPU_LAYERS" \
  --ctx-size "$CTX_SIZE" \
  --threads "$THREADS" \
  --parallel "$PARALLEL" \
  --jinja \
  --reasoning "$REASONING" \
  -a "$SERVED_NAME" \
  $LLAMA_EXTRA_ARGS > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Wait for server readiness
ready=0
for attempt in $(seq 1 180); do
  sleep 5
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[q4km-9b] server exited before ready" >&2
    tail -120 "$SERVER_LOG" || true
    exit 4
  fi
  smoke="$(curl -s -m 30 -H 'Content-Type: application/json' \
    -d '{"model":"'"$SERVED_NAME"'","prompt":"A","max_tokens":4,"temperature":0}' \
    "http://${HOST}:${PORT}/v1/completions" 2>&1 || true)"
  if echo "$smoke" | grep -q '"choices"'; then
    ready=1
    break
  fi
  if (( attempt % 12 == 0 )); then
    echo "[q4km-9b] waiting for server... attempt $attempt/180"
  fi
done
if [[ "$ready" != "1" ]]; then
  echo "[q4km-9b] server not ready after 900s" >&2
  tail -120 "$SERVER_LOG" || true
  exit 5
fi

echo "[q4km-9b] server ready"

# ---------------------------------------------------------------------------
# If openai_serving_matrix_probe.py exists, use it
# ---------------------------------------------------------------------------
MATRIX_PROBE="$REPO_ROOT/benchmarks/openai_serving_matrix_probe.py"
if [[ -f "$MATRIX_PROBE" ]]; then
  echo "[q4km-9b] running matrix probe..."
  "$PY" "$MATRIX_PROBE" \
    --url "http://${HOST}:${PORT}/v1" \
    --model "$SERVED_NAME" \
    --single-max-tokens $SINGLE_MAX_TOKENS \
    --concurrency $CONCURRENCY \
    --concurrent-max-tokens "$CONCURRENT_MAX_TOKENS" \
    --long-context-chars $LONG_CONTEXT_CHARS \
    --long-context-max-tokens "$LONG_CONTEXT_MAX_TOKENS" \
    --runs 1 \
    --timeout 1800 \
    --out "$OUT"
else
  # ---------------------------------------------------------------------------
  # Fallback: inline OpenAI API probe
  # ---------------------------------------------------------------------------
  echo "[q4km-9b] matrix probe not found, using inline API probe"

  python3 - "$OUT" "$HOST" "$PORT" "$SERVED_NAME" \
      "$SINGLE_MAX_TOKENS" "$CONCURRENCY" "$CONCURRENT_MAX_TOKENS" \
      "$LONG_CONTEXT_CHARS" "$LONG_CONTEXT_MAX_TOKENS" \
      "$GGUF_FOUND" "$GGUF_SIZE_GIB" "$LLAMA_SERVER" "$LLAMA_GIT_REV" <<'PYEOF'
import json, sys, time, urllib.request, urllib.error, statistics

out_path = sys.argv[1]
host, port, model = sys.argv[2], sys.argv[3], sys.argv[4]
single_tokens = sys.argv[5].split()
concurrency_str = sys.argv[6].split()
concurrent_max = int(sys.argv[7])
long_chars_str = sys.argv[8].split()
long_max = int(sys.argv[9])
gguf_path = sys.argv[10]
gguf_size = sys.argv[11]
llama_bin = sys.argv[12]
git_rev = sys.argv[13]

base = f"http://{host}:{port}/v1"

def chat_complete(prompt, max_tokens, timeout=600):
    """Call /v1/chat/completions and return (tps, prompt_tokens, error)."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        elapsed = time.monotonic() - t0
        comp = data["choices"][0]["message"]["content"]
        pt = data.get("usage", {}).get("prompt_tokens", 0)
        ct = data.get("usage", {}).get("completion_tokens", len(comp.split()))
        tps = ct / elapsed if elapsed > 0 else 0
        return tps, pt, ct, elapsed, None
    except Exception as e:
        return 0, 0, 0, 0, str(e)

# --- Single TPS ---
single_results = {}
for mt_str in single_tokens:
    mt = int(mt_str)
    prompt = "Explain the theory of relativity in detail. " * 20
    print(f"  [single] max_tokens={mt}...")
    tps, pt, ct, elapsed, err = chat_complete(prompt, mt)
    single_results[mt_str] = {
        "ok": err is None,
        "wall_tps": round(tps, 2),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "elapsed_s": round(elapsed, 3),
        "error": err,
    }
    print(f"    → {tps:.1f} TPS, {pt} prompt tok, err={err}")

# --- Concurrent TPS ---
import concurrent.futures
concurrent_results = {}
for c_str in concurrency_str:
    c = int(c_str)
    prompt = "Write a short story about a robot. " * 10
    print(f"  [concurrent] n={c}...")
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
        futs = [pool.submit(chat_complete, prompt, concurrent_max, 600) for _ in range(c)]
        results = [f.result() for f in futs]
    elapsed = time.monotonic() - t0
    total_tps = sum(r[0] for r in results)
    errors = [r[4] for r in results if r[4]]
    concurrent_results[c_str] = {
        "ok": len(errors) == 0,
        "batch_wall_tps": round(total_tps, 2),
        "elapsed_s": round(elapsed, 3),
        "errors": errors[:2],
    }
    print(f"    → {total_tps:.1f} total TPS, errors={errors[:2]}")

# --- Long Context ---
long_results = {}
for lc_str in long_chars_str:
    lc = int(lc_str)
    # Build ~lc chars of context
    chunk = "The quick brown fox jumps over the lazy dog. " * 100
    repeats = (lc // len(chunk)) + 1
    long_prompt = (chunk * repeats)[:lc]
    print(f"  [long_context] chars={lc}...")
    tps, pt, ct, elapsed, err = chat_complete(long_prompt, long_max, timeout=300)
    long_results[lc_str] = {
        "ok": err is None,
        "prompt_chars": lc,
        "prompt_tokens": pt,
        "wall_tps": round(tps, 2),
        "completion_tokens": ct,
        "elapsed_s": round(elapsed, 3),
        "error": err,
    }
    status = f"{tps:.1f} TPS" if err is None else f"ERR: {err}"
    print(f"    → {status}")

report = {
    "schema": "lynn-qwen35-9b-q4km-baseline-v1",
    "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "status": "DONE",
    "model_id": "Qwen3.5-9B",
    "quant": "Q4_K_M",
    "model_path": gguf_path,
    "size_gib": gguf_size,
    "llama_cpp_binary": llama_bin,
    "git_rev": git_rev,
    "single_tps": single_results,
    "concurrent_tps": concurrent_results,
    "long_context": long_results,
    "errors": [],
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n[q4km-9b] Report written: {out_path}")
PYEOF
fi

# ---------------------------------------------------------------------------
# Optional quality evaluation (QUALITY=1)
# ---------------------------------------------------------------------------
if [[ "$QUALITY" == "1" ]]; then
  echo "[q4km-9b] running quality evaluations..."
  QUALITY_PREFIX="$REPORT_ROOT/q4km_llamacpp_${STAMP}"
  MMLU_SCRIPT="$REPO_ROOT/scripts/openai_mmlu_500_5shot_eval.py"
  GPQA_SCRIPT="$REPO_ROOT/scripts/openai_gpqa_diamond_eval.py"
  if [[ -f "$MMLU_SCRIPT" && -d "$MMLU_DATA_DIR" ]]; then
    echo "[q4km-9b] MMLU 500 5-shot..."
    "$PY" "$MMLU_SCRIPT" \
      --data-dir "$MMLU_DATA_DIR" \
      --base-url "http://${HOST}:${PORT}/v1" \
      --model "$SERVED_NAME" \
      --out "${QUALITY_PREFIX}_mmlu_n500.jsonl" \
      --concurrency 6 \
      --shots 5 \
      --sample 500 \
      --timeout 120
  else
    echo "[q4km-9b] MMLU skipped: script=$MMLU_SCRIPT data_dir=$MMLU_DATA_DIR"
  fi
  if [[ -f "$GPQA_SCRIPT" && -f "$GPQA_CSV" ]]; then
    echo "[q4km-9b] GPQA Diamond..."
    "$PY" "$GPQA_SCRIPT" \
      --csv "$GPQA_CSV" \
      --base-url "http://${HOST}:${PORT}/v1" \
      --model "$SERVED_NAME" \
      --out "${QUALITY_PREFIX}_gpqa.jsonl" \
      --concurrency 2 \
      --timeout 120
  else
    echo "[q4km-9b] GPQA skipped: script=$GPQA_SCRIPT csv=$GPQA_CSV"
  fi
fi

# ---------------------------------------------------------------------------
# Post-process: summary
# ---------------------------------------------------------------------------
echo "[q4km-9b] generating summary..."
"$PY" "$REPO_ROOT/scripts/summarize_qwen35_9b_q4km_baseline.py" \
  --report "$OUT" \
  --output "$REPORT_ROOT/r6000_qwen35_9b_q4km_baseline_${STAMP}.md" \
  2>/dev/null || echo "[q4km-9b] summary generation failed (non-fatal)"

echo "[q4km-9b] done. reports in $REPORT_ROOT/"
