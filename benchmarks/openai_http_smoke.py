#!/usr/bin/env python3
"""OpenAI-compatible smoke test for Lynn engine HTTP servers.

This keeps serving validation reproducible. It checks health, model listing,
chat completions, and text completions, then writes a JSON report suitable for
Phase 4 milestone docs and CI-style gates.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib import request


def _json_request(method: str, url: str, payload: dict[str, Any] | None = None) -> tuple[dict[str, Any], float]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, headers=headers, method=method)
    t0 = time.time()
    with request.urlopen(req, timeout=300) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body), time.time() - t0


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:18200/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    base = args.url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    checks: list[dict[str, Any]] = []

    health, health_s = _json_request("GET", f"{root}/health")
    checks.append({
        "name": "health",
        "ok": health.get("status") == "ok",
        "elapsed_s": health_s,
        "response": health,
    })

    models, models_s = _json_request("GET", f"{base}/models")
    checks.append({
        "name": "models",
        "ok": isinstance(models.get("data"), list) and len(models["data"]) >= 1,
        "elapsed_s": models_s,
        "response": models,
    })

    chat_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "用一句话解释MoE active parameters"}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }
    chat, chat_s = _json_request("POST", f"{base}/chat/completions", chat_payload)
    chat_content = (
        chat.get("choices", [{}])[0]
        .get("message", {})
        .get("content")
    )
    checks.append({
        "name": "chat_completions",
        "ok": _nonempty(chat_content),
        "elapsed_s": chat_s,
        "content": chat_content,
        "usage": chat.get("usage"),
        "metrics": chat.get("_lynn_engine_metrics"),
    })

    completion_payload = {
        "model": args.model,
        "prompt": "Python递归阶乘函数:",
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }
    completion, completion_s = _json_request("POST", f"{base}/completions", completion_payload)
    text = completion.get("choices", [{}])[0].get("text")
    checks.append({
        "name": "completions",
        "ok": _nonempty(text),
        "elapsed_s": completion_s,
        "text": text,
        "usage": completion.get("usage"),
        "metrics": completion.get("_lynn_engine_metrics"),
    })

    result = {
        "schema_version": "lynn-engine-openai-http-smoke-v1",
        "url": base,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "checks": checks,
    }
    result["verdict"] = "PASS" if all(c["ok"] for c in checks) else "FAIL"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
