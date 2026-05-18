#!/usr/bin/env python3
"""P25: measure the OpenAI server decode path with Lynn metrics.

`benchmarks/throughput_bench.py` is framework-generic and defaults to
temperature=0.7. Lynn engine intentionally only serves greedy temperature=0
today, and its HTTP responses expose `_lynn_engine_metrics.timings.decode_tps`.

This probe is the server-side counterpart to the internal graph benchmarks: it
hits `/v1/completions`, keeps temperature fixed at 0, and records both wall TPS
and engine decode TPS. It is deliberately small so we can run it on R6000/Spark
without pulling in external benchmarking dependencies.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
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
        data = json.load(resp)
    return data, time.time() - t0


def _run_one(
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    chat: bool,
) -> dict[str, Any]:
    if chat:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        endpoint = "/chat/completions"
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        endpoint = "/completions"
    data, wall_s = _post_json(url, endpoint, payload, timeout)
    usage = data.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens") or 0)
    metrics = data.get("_lynn_engine_metrics", {})
    timings = metrics.get("timings", {})
    decode_steps = timings.get("decode_step_seconds") or []
    choice = (data.get("choices") or [{}])[0]
    if chat:
        preview = (choice.get("message") or {}).get("content", "")[:160]
    else:
        preview = choice.get("text", "")[:160]
    return {
        "wall_s": wall_s,
        "completion_tokens": completion_tokens,
        "wall_tps": completion_tokens / wall_s if wall_s else None,
        "metrics_tokens_per_second": metrics.get("tokens_per_second"),
        "decode_tps": timings.get("decode_tps"),
        "prefill_seconds": timings.get("prefill_seconds"),
        "linear_block_graph_reused": timings.get("linear_block_graph_reused"),
        "linear_block_graph_capture_seconds": timings.get("linear_block_graph_capture_seconds"),
        "linear_block_graph_prewarm_seconds": timings.get("linear_block_graph_prewarm_seconds"),
        "native_fp4_lm_head_enabled": timings.get("native_fp4_lm_head_enabled"),
        "decode_step_count": len(decode_steps),
        "decode_step_ms_mean": (statistics.mean(decode_steps) * 1000.0) if decode_steps else None,
        "decode_step_ms_median": (statistics.median(decode_steps) * 1000.0) if decode_steps else None,
        "preview": preview,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    out: dict[str, Any] = {"runs": len(rows)}
    for key in ("wall_tps", "metrics_tokens_per_second", "decode_tps", "prefill_seconds", "decode_step_ms_median"):
        xs = vals(key)
        if xs:
            out[key] = {
                "mean": statistics.mean(xs),
                "median": statistics.median(xs),
                "min": min(xs),
                "max": max(xs),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="OpenAI-compatible base URL ending in /v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--chat", action="store_true", help="Use /v1/chat/completions with a user message.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results: list[dict[str, Any]] = []
    for max_tokens in args.max_tokens:
        for run_idx in range(args.runs):
            row = _run_one(
                url=args.url,
                model=args.model,
                prompt=args.prompt,
                max_tokens=max_tokens,
                timeout=args.timeout,
                chat=args.chat,
            )
            row["max_tokens"] = max_tokens
            row["run_idx"] = run_idx
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    by_tokens: dict[str, Any] = {}
    for max_tokens in args.max_tokens:
        rows = [r for r in results if r["max_tokens"] == max_tokens]
        by_tokens[str(max_tokens)] = _summary(rows)

    report = {
        "schema_version": "lynn-engine-p25-server-decode-tps-probe-v1",
        "url": args.url,
        "model": args.model,
        "chat": args.chat,
        "prompt_chars": len(args.prompt),
        "results": results,
        "summary_by_max_tokens": by_tokens,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
