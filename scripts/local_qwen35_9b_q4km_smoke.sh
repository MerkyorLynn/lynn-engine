#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Smoke test for the local Qwen3.5-9B Q4_K_M llama.cpp endpoint (V2).
#
# Tests:
#   1. Health check (fail-fast if server not running)
#   2. Chinese short answer (/v1/chat/completions)
#   3. JSON object structured output (response_format=json_object)
#
# Usage:
#   bash scripts/local_qwen35_9b_q4km_smoke.sh
#   BASE_URL=http://127.0.0.1:8080/v1 bash scripts/local_qwen35_9b_q4km_smoke.sh
#   bash scripts/local_qwen35_9b_q4km_smoke.sh --dry-run
#
# Options:
#   --dry-run     Validate script logic without connecting (offline check)
#   --port PORT   Override port (default: 18099)
#   --out FILE    Write JSON report to file
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL="${BASE_URL:-http://127.0.0.1:18099/v1}"
MODEL="${MODEL:-qwen35-9b-q4km}"
OUT_JSON="${OUT_JSON:-}"
TIMEOUT="${TIMEOUT:-120}"
DRY_RUN="${DRY_RUN:-0}"

# CLI flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)     BASE_URL="http://127.0.0.1:$2/v1"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model)    MODEL="$2"; shift 2 ;;
    --out)      OUT_JSON="$2"; shift 2 ;;
    --timeout)  TIMEOUT="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --help|-h)
      echo "Usage: $0 [--port PORT] [--base-url URL] [--model NAME] [--out FILE] [--dry-run]"
      exit 0
      ;;
    *) echo "[smoke] Unknown flag: $1" >&2; exit 1 ;;
  esac
done

# ─────────────────────────────────────────────────────────────────────────────
# DRY_RUN mode: validate script, show config, exit
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[smoke] DRY_RUN=1 — offline validation mode"
  echo "[smoke] Target:  $BASE_URL"
  echo "[smoke] Model:   $MODEL"
  echo "[smoke] Timeout: ${TIMEOUT}s"
  echo "[smoke] Out:     ${OUT_JSON:-<stdout only>}"
  echo ""
  echo "[smoke] Script syntax: OK (bash -n passed if you're seeing this)"
  echo "[smoke] Python check:"
  python3 -c "import json, urllib.request, sys; print(f'  python3={sys.executable} version={sys.version_info.major}.{sys.version_info.minor}')"
  echo ""
  echo "[smoke] DRY_RUN complete. To run for real, remove --dry-run flag."
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Health check (fail-fast)
# ─────────────────────────────────────────────────────────────────────────────
HEALTH_URL="${BASE_URL%/v1}/health"
echo "[smoke] Checking health: $HEALTH_URL"

HTTP_CODE="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 "$HEALTH_URL" 2>/dev/null)" || HTTP_CODE="000"

if [[ "$HTTP_CODE" != "200" ]]; then
  cat >&2 <<EOF
[smoke] ERROR: Server not responding (HTTP $HTTP_CODE)

  Health URL: $HEALTH_URL

  Is the server running? Start it with:
    bash scripts/local_qwen35_9b_q4km_llamacpp_server.sh

  Or check a different port:
    bash scripts/local_qwen35_9b_q4km_smoke.sh --port 8080
EOF
  exit 1
fi
echo "[smoke] Health: OK (HTTP 200)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Run test suite via embedded Python (no external deps beyond stdlib)
# ─────────────────────────────────────────────────────────────────────────────
python3 - "$BASE_URL" "$MODEL" "$OUT_JSON" "$TIMEOUT" <<'PY'
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path


base_url, model, out_json, timeout_raw = sys.argv[1:5]
timeout = float(timeout_raw)


def post_chat(messages, max_tokens=96, response_format=None):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if response_format:
        payload["response_format"] = response_format
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
    elapsed = time.time() - t0
    data = json.loads(raw)
    text = data["choices"][0]["message"]["content"]
    return {"elapsed_seconds": elapsed, "text": text, "response": data}


rows = []
ok = True
errors = []

# Test 1: Chinese short answer
print("[smoke] Test 1: Chinese short answer...")
try:
    short = post_chat(
        [{"role": "user", "content": "用一句话说明本地智能体的价值。"}],
        max_tokens=64,
    )
    short_ok = bool(short["text"].strip())
    rows.append({"id": "short_zh", "ok": short_ok, **short})
    ok = ok and short_ok
    if short_ok:
        print(f"  PASS: {short['text'][:80]}")
    else:
        errors.append("short_zh returned empty text")
        print("  FAIL: empty response")
except Exception as exc:  # noqa: BLE001 - command-line diagnostic
    rows.append({"id": "short_zh", "ok": False, "error": str(exc)})
    ok = False
    errors.append(f"short_zh error: {exc}")
    print(f"  FAIL: {exc}")

# Test 2: JSON structured output
print("[smoke] Test 2: JSON object output...")
try:
    structured = post_chat(
        [
            {"role": "system", "content": "You output JSON only. No markdown."},
            {"role": "user", "content": 'Return one JSON object with keys model and runtime. Use model="qwen35-9b" and runtime="llama.cpp".'},
        ],
        max_tokens=96,
        response_format={"type": "json_object"},
    )
    text = structured["text"].strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    parsed = json.loads(text)
    structured_ok = (
        isinstance(parsed, dict)
        and "qwen35-9b" in json.dumps(parsed, ensure_ascii=False)
        and "llama.cpp" in json.dumps(parsed, ensure_ascii=False)
    )
    rows.append({"id": "json_runtime", "ok": structured_ok, "parsed": parsed, **structured})
    ok = ok and structured_ok
    if structured_ok:
        print(f"  PASS: {json.dumps(parsed)}")
    else:
        errors.append("json_runtime failed content check")
        print(f"  FAIL: content check failed: {text[:100]}")
except json.JSONDecodeError as exc:
    rows.append({"id": "json_runtime", "ok": False, "error": f"JSON parse: {exc}"})
    ok = False
    errors.append(f"json_runtime JSON parse error: {exc}")
    print(f"  FAIL: JSON parse error: {exc}")
except Exception as exc:  # noqa: BLE001 - command-line diagnostic
    rows.append({"id": "json_runtime", "ok": False, "error": str(exc)})
    ok = False
    errors.append(f"json_runtime error: {exc}")
    print(f"  FAIL: {exc}")

# Report
report = {
    "schema": "lynn-qwen35-9b-q4km-local-smoke-v2",
    "base_url": base_url,
    "model": model,
    "ok": ok,
    "errors": errors,
    "rows": rows,
}
if out_json:
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[smoke] Report written: {out_json}")

print("\n" + "─" * 60)
summary = json.dumps({
    "ok": ok,
    "base_url": base_url,
    "model": model,
    "errors": errors,
    "texts": {row["id"]: (row.get("text") or "")[:160] for row in rows},
}, ensure_ascii=False, indent=2)
print(summary)

if ok:
    print("\n[smoke] ALL TESTS PASSED")
else:
    print("\n[smoke] SOME TESTS FAILED")
raise SystemExit(0 if ok else 2)
PY
