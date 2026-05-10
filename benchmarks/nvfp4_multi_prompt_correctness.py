#!/usr/bin/env python3
"""
Lynn Engine NVFP4 multi-prompt correctness harness.

Compares a baseline OpenAI-compatible completion endpoint against a candidate
endpoint (for example a 27B/NVFP4 server) across a diverse prompt set. This is a
gate script: any token mismatch is a failure. Do not use single-prompt results to
approve default runtime changes.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS = [
    "The capital of France is",
    "What is the speed of light",
    "用一句话解释什么是 transformer",
    "def fibonacci(n):",
    "2+2=",
    "Python is",
    "Write a haiku about spring",
    "今天天气",
    "import torch",
    "Hello world",
    "The largest planet is",
    "I love eating",
    "The quick brown fox",
    "import numpy as",
    "上海今天适合出门吗?",
    "请用三句话解释 MoE 模型里的 active parameters",
    "function debounce(fn, wait) {",
    "AAPL stock price",
    "把 I love Beijing Tiananmen Square 翻译成中文",
    "Once upon a time in a small village,",
]


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def completion(base_url: str, model: str, prompt: str, *, max_tokens: int, timeout: int) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "logprobs": 10,
    }
    t0 = time.time()
    data = post_json(url, payload, timeout)
    elapsed = time.time() - t0
    choice = data["choices"][0]
    token_ids = choice.get("token_ids")
    if token_ids is None:
        token_ids = data.get("usage", {}).get("completion_tokens_details", {}).get("token_ids")
    return {
        "text": choice.get("text", ""),
        "token_ids": token_ids,
        "logprobs": choice.get("logprobs"),
        "elapsed_s": elapsed,
        "raw_id": data.get("id"),
    }


def load_prompts(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        return [str(x["prompt"] if isinstance(x, dict) else x) for x in data]
    prompts: list[str] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            obj = json.loads(line)
            prompts.append(str(obj.get("prompt", obj.get("text", line))))
        else:
            prompts.append(line)
    return prompts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:18002/v1")
    ap.add_argument("--baseline-model", required=True)
    ap.add_argument("--candidate-url", required=True, help="OpenAI-compatible base URL for NVFP4 candidate")
    ap.add_argument("--candidate-model", required=True)
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default="/root/autodl-tmp/results/nvfp4_multi_prompt_correctness.json")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    results: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    print(f"NVFP4 correctness gate: {len(prompts)} prompts x {args.max_tokens} tokens")
    for idx, prompt in enumerate(prompts, 1):
        print(f"\n[{idx}/{len(prompts)}] {prompt!r}", flush=True)
        base = completion(args.baseline_url, args.baseline_model, prompt, max_tokens=args.max_tokens, timeout=args.timeout)
        cand = completion(args.candidate_url, args.candidate_model, prompt, max_tokens=args.max_tokens, timeout=args.timeout)
        base_key = base["token_ids"] if base["token_ids"] is not None else base["text"]
        cand_key = cand["token_ids"] if cand["token_ids"] is not None else cand["text"]
        ok = base_key == cand_key
        rec = {
            "prompt": prompt,
            "ok": ok,
            "baseline": {"token_ids": base["token_ids"], "text": base["text"], "elapsed_s": round(base["elapsed_s"], 3)},
            "candidate": {"token_ids": cand["token_ids"], "text": cand["text"], "elapsed_s": round(cand["elapsed_s"], 3)},
        }
        results.append(rec)
        if not ok:
            mismatches.append(rec)
        print(f"  {'PASS' if ok else 'FAIL'} base={base_key!r} candidate={cand_key!r}", flush=True)

    exact = (len(results) - len(mismatches)) / len(results) if results else 0.0
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "baseline_url": args.baseline_url,
        "baseline_model": args.baseline_model,
        "candidate_url": args.candidate_url,
        "candidate_model": args.candidate_model,
        "n_prompts": len(results),
        "max_tokens": args.max_tokens,
        "exact_match_rate": exact,
        "mismatches_count": len(mismatches),
        "verdict": "PASS" if exact == 1.0 and not mismatches else "FAIL",
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved {out}")
    print(f"verdict={report['verdict']} exact_match_rate={exact:.3f} mismatches={len(mismatches)}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
