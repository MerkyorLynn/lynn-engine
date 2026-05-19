#!/usr/bin/env bash
set -euo pipefail

# Smoke test for the local Qwen3.5-9B Q4_K_M llama.cpp endpoint.

BASE_URL="${BASE_URL:-http://127.0.0.1:18099/v1}"
MODEL="${MODEL:-qwen35-9b-q4km}"
OUT_JSON="${OUT_JSON:-}"
TIMEOUT="${TIMEOUT:-120}"

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

try:
    short = post_chat(
        [{"role": "user", "content": "用一句话说明本地智能体的价值。"}],
        max_tokens=64,
    )
    short_ok = bool(short["text"].strip())
    rows.append({"id": "short_zh", "ok": short_ok, **short})
    ok = ok and short_ok
    if not short_ok:
        errors.append("short_zh returned empty text")
except Exception as exc:  # noqa: BLE001 - command-line diagnostic
    rows.append({"id": "short_zh", "ok": False, "error": str(exc)})
    ok = False
    errors.append(f"short_zh error: {exc}")

try:
    structured = post_chat(
        [
            {"role": "system", "content": "You output JSON only. No markdown."},
            {"role": "user", "content": "Return one JSON object with keys model and runtime. Use model=\"qwen35-9b\" and runtime=\"llama.cpp\"."},
        ],
        max_tokens=96,
        response_format={"type": "json_object"},
    )
    text = structured["text"].strip()
    parsed = json.loads(text)
    structured_ok = (
        isinstance(parsed, dict)
        and "qwen35-9b" in json.dumps(parsed, ensure_ascii=False)
        and "llama.cpp" in json.dumps(parsed, ensure_ascii=False)
    )
    rows.append({"id": "json_runtime", "ok": structured_ok, "parsed": parsed, **structured})
    ok = ok and structured_ok
    if not structured_ok:
        errors.append("json_runtime failed content check")
except Exception as exc:  # noqa: BLE001 - command-line diagnostic
    rows.append({"id": "json_runtime", "ok": False, "error": str(exc)})
    ok = False
    errors.append(f"json_runtime error: {exc}")

report = {
    "schema": "lynn-qwen35-9b-q4km-local-smoke-v1",
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
print(json.dumps({
    "ok": ok,
    "base_url": base_url,
    "model": model,
    "errors": errors,
    "texts": {row["id"]: (row.get("text") or "")[:160] for row in rows},
}, ensure_ascii=False, indent=2))
raise SystemExit(0 if ok else 2)
PY
