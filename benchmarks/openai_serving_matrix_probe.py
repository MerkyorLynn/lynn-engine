#!/usr/bin/env python3
"""Generic OpenAI-compatible serving matrix probe.

This is intentionally framework-neutral. Lynn Engine responses may expose
internal decode metrics, but llama.cpp only gives OpenAI-style usage counts, so
this probe measures wall-clock throughput for single-stream, concurrent, and
long-context requests using the same HTTP surface.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "请连续输出一段关于 MoE 推理优化、NVFP4、CUDA graph 和工具调用服务化的中文技术说明。"
    "要求持续展开，不要提前结束。"
)


def _post_json(url: str, endpoint: str, payload: dict[str, Any], timeout: int) -> tuple[dict[str, Any], float]:
    req = urllib.request.Request(
        url.rstrip("/") + endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp), time.time() - t0


def _completion_request(
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    try:
        data, wall_s = _post_json(url, "/completions", payload, timeout)
        usage = data.get("usage", {})
        completion_tokens = int(usage.get("completion_tokens") or 0)
        prompt_tokens = usage.get("prompt_tokens")
        choice = (data.get("choices") or [{}])[0]
        return {
            "ok": True,
            "wall_s": wall_s,
            "prompt_chars": len(prompt),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "wall_tps": completion_tokens / wall_s if wall_s else None,
            "preview": (choice.get("text") or "")[:160],
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, http.client.HTTPException) as exc:
        return {
            "ok": False,
            "wall_s": None,
            "prompt_chars": len(prompt),
            "prompt_tokens": None,
            "completion_tokens": 0,
            "wall_tps": None,
            "error": repr(exc),
        }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in rows if r.get("ok")]
    tps = [float(r["wall_tps"]) for r in ok_rows if r.get("wall_tps") is not None]
    tokens = sum(int(r.get("completion_tokens") or 0) for r in ok_rows)
    wall_sum = sum(float(r.get("wall_s") or 0) for r in ok_rows)
    out: dict[str, Any] = {
        "count": len(rows),
        "ok": len(ok_rows),
        "failed": len(rows) - len(ok_rows),
        "completion_tokens_total": tokens,
        "wall_seconds_sum": wall_sum,
    }
    if tps:
        out["wall_tps"] = {
            "mean": statistics.mean(tps),
            "median": statistics.median(tps),
            "min": min(tps),
            "max": max(tps),
        }
    return out


def _concurrency_batch_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("concurrency")), []).append(row)

    out: dict[str, Any] = {}
    for concurrency, group_rows in grouped.items():
        first = group_rows[0]
        ok = sum(1 for row in group_rows if row.get("ok"))
        out[concurrency] = {
            "requests": len(group_rows),
            "ok": ok,
            "batch_wall_s": first.get("batch_wall_s"),
            "batch_completion_tokens": first.get("batch_completion_tokens"),
            "batch_wall_tps": first.get("batch_wall_tps"),
        }
    return out


def _run_single(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for max_tokens in args.single_max_tokens:
        for run_idx in range(args.runs):
            row = _completion_request(
                url=args.url,
                model=args.model,
                prompt=args.prompt,
                max_tokens=max_tokens,
                timeout=args.timeout,
            )
            row.update({"kind": "single", "max_tokens": max_tokens, "run_idx": run_idx})
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def _run_concurrency(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concurrency in args.concurrency:
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(
                    _completion_request,
                    url=args.url,
                    model=args.model,
                    prompt=args.prompt,
                    max_tokens=args.concurrent_max_tokens,
                    timeout=args.timeout,
                )
                for _ in range(concurrency)
            ]
            batch_rows = [f.result() for f in futures]
        batch_wall_s = time.time() - t0
        completion_tokens = sum(int(r.get("completion_tokens") or 0) for r in batch_rows if r.get("ok"))
        for idx, row in enumerate(batch_rows):
            row.update(
                {
                    "kind": "concurrency",
                    "concurrency": concurrency,
                    "request_idx": idx,
                    "max_tokens": args.concurrent_max_tokens,
                    "batch_wall_s": batch_wall_s,
                    "batch_completion_tokens": completion_tokens,
                    "batch_wall_tps": completion_tokens / batch_wall_s if batch_wall_s else None,
                }
            )
            rows.append(row)
        print(
            json.dumps(
                {
                    "kind": "concurrency_summary",
                    "concurrency": concurrency,
                    "ok": sum(1 for r in batch_rows if r.get("ok")),
                    "batch_wall_s": batch_wall_s,
                    "batch_completion_tokens": completion_tokens,
                    "batch_wall_tps": completion_tokens / batch_wall_s if batch_wall_s else None,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return rows


def _long_prompt(chars: int) -> str:
    seed = (
        "Lynn Engine long-context benchmark paragraph. "
        "This sentence discusses MoE routing, KV cache pressure, linear attention, "
        "tool-call JSON formatting, and CUDA graph replay stability. "
    )
    repeats = max(1, chars // len(seed) + 1)
    return (seed * repeats)[:chars] + "\n\n请用中文总结上文的核心技术挑战，持续输出。"


def _run_long_context(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chars in args.long_context_chars:
        prompt = _long_prompt(chars)
        row = _completion_request(
            url=args.url,
            model=args.model,
            prompt=prompt,
            max_tokens=args.long_context_max_tokens,
            timeout=args.timeout,
        )
        row.update(
            {
                "kind": "long_context",
                "target_prompt_chars": chars,
                "max_tokens": args.long_context_max_tokens,
            }
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="OpenAI-compatible base URL ending in /v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--single-max-tokens", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--concurrency", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--concurrent-max-tokens", type=int, default=256)
    ap.add_argument("--long-context-chars", type=int, nargs="+", default=[8192, 32768, 65536])
    ap.add_argument("--long-context-max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    single = _run_single(args)
    concurrency = _run_concurrency(args)
    long_context = _run_long_context(args)
    report = {
        "schema_version": "openai-serving-matrix-probe-v1",
        "url": args.url,
        "model": args.model,
        "single": {"rows": single, "summary": _summary(single)},
        "concurrency": {
            "rows": concurrency,
            "summary": _summary(concurrency),
            "batch_summary": _concurrency_batch_summary(concurrency),
        },
        "long_context": {"rows": long_context, "summary": _summary(long_context)},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
