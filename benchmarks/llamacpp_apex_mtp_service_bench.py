#!/usr/bin/env python3
"""Small OpenAI-compatible benchmark for the Spark llama.cpp APEX-MTP service.

The script is intentionally HTTP-only: it does not start a model process and is
safe to run against the already-loaded fallback service.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROMPTS = [
    "Answer directly in one paragraph: why does speculative decoding help LLM serving?",
    "Write a compact Python function that returns the first n Fibonacci numbers.",
    "Summarize the tradeoff between latency and throughput for batched decoding.",
    "Give three concise debugging steps for a CUDA kernel returning wrong values.",
]


@dataclass
class RequestResult:
    ok: bool
    label: str
    latency_s: float
    status: int | None
    error: str | None
    response: dict[str, Any] | None

    @property
    def completion_tokens(self) -> int:
        if not self.response:
            return 0
        usage = self.response.get("usage") or {}
        timings = self.response.get("timings") or {}
        return int(timings.get("predicted_n") or usage.get("completion_tokens") or 0)

    @property
    def prompt_tokens(self) -> int:
        if not self.response:
            return 0
        usage = self.response.get("usage") or {}
        timings = self.response.get("timings") or {}
        return int(timings.get("prompt_n") or usage.get("prompt_tokens") or 0)

    @property
    def draft_n(self) -> int:
        if not self.response:
            return 0
        return int((self.response.get("timings") or {}).get("draft_n") or 0)

    @property
    def draft_n_accepted(self) -> int:
        if not self.response:
            return 0
        return int((self.response.get("timings") or {}).get("draft_n_accepted") or 0)

    @property
    def server_tps(self) -> float:
        if not self.response:
            return 0.0
        timings = self.response.get("timings") or {}
        return float(timings.get("predicted_per_second") or 0.0)


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


def get_text(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def run_one(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    label: str,
    speculative_n_max: int | None,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "timings_per_token": True,
    }
    if speculative_n_max is not None:
        payload["speculative.n_max"] = speculative_n_max

    t0 = time.perf_counter()
    try:
        status, response = post_json(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            payload,
            timeout,
        )
        latency = time.perf_counter() - t0
        return RequestResult(True, label, latency, status, None, response)
    except urllib.error.HTTPError as exc:
        latency = time.perf_counter() - t0
        body = exc.read().decode("utf-8", errors="replace")
        return RequestResult(False, label, latency, exc.code, body[:2000], None)
    except Exception as exc:  # noqa: BLE001 - benchmark should preserve failures.
        latency = time.perf_counter() - t0
        return RequestResult(False, label, latency, None, repr(exc), None)


def summarize(name: str, results: list[RequestResult], wall_s: float) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    tps_values = [r.server_tps for r in ok if r.server_tps > 0]
    latency_values = [r.latency_s for r in ok]
    draft_total = sum(r.draft_n for r in ok)
    draft_accepted = sum(r.draft_n_accepted for r in ok)
    completion_total = sum(r.completion_tokens for r in ok)

    return {
        "name": name,
        "requests": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "wall_s": wall_s,
        "aggregate_completion_tokens": completion_total,
        "aggregate_wall_tps": completion_total / wall_s if wall_s > 0 else 0.0,
        "server_tps_mean": statistics.fmean(tps_values) if tps_values else 0.0,
        "server_tps_median": statistics.median(tps_values) if tps_values else 0.0,
        "latency_s_mean": statistics.fmean(latency_values) if latency_values else 0.0,
        "latency_s_median": statistics.median(latency_values) if latency_values else 0.0,
        "draft_n": draft_total,
        "draft_n_accepted": draft_accepted,
        "draft_acceptance": draft_accepted / draft_total if draft_total else 0.0,
        "errors": [r.error for r in results if not r.ok],
    }


def as_dict(result: RequestResult) -> dict[str, Any]:
    response = result.response or {}
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {
        "ok": result.ok,
        "label": result.label,
        "latency_s": result.latency_s,
        "status": result.status,
        "error": result.error,
        "completion_tokens": result.completion_tokens,
        "prompt_tokens": result.prompt_tokens,
        "server_tps": result.server_tps,
        "draft_n": result.draft_n,
        "draft_n_accepted": result.draft_n_accepted,
        "finish_reason": choice.get("finish_reason"),
        "content_preview": (message.get("content") or message.get("reasoning_content") or "")[:240],
        "timings": response.get("timings"),
        "usage": response.get("usage"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18098")
    parser.add_argument("--model", default="qwen36-35b-a3b-apex-mtp")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--single-runs", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--concurrent-rounds", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--speculative-n-max", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = args.output or Path(f"reports/mtp/llamacpp_apex_mtp_service_bench_{run_id}.json")

    health = None
    slots = None
    try:
        health = get_text(f"{args.base_url.rstrip('/')}/health", args.timeout)
    except Exception as exc:  # noqa: BLE001
        health = f"ERROR: {exc!r}"
    try:
        slots = json.loads(get_text(f"{args.base_url.rstrip('/')}/slots", args.timeout))
    except Exception as exc:  # noqa: BLE001
        slots = f"ERROR: {exc!r}"

    single_results: list[RequestResult] = []
    t0 = time.perf_counter()
    for i in range(args.single_runs):
        prompt = PROMPTS[i % len(PROMPTS)]
        single_results.append(
            run_one(
                args.base_url,
                args.model,
                prompt,
                args.max_tokens,
                args.temperature,
                args.timeout,
                f"single-{i}",
                args.speculative_n_max,
            )
        )
    single_wall = time.perf_counter() - t0

    concurrent_jobs = []
    for r in range(args.concurrent_rounds):
        for i in range(args.concurrency):
            concurrent_jobs.append((r, i, PROMPTS[(r * args.concurrency + i) % len(PROMPTS)]))

    concurrent_results: list[RequestResult] = []
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [
            pool.submit(
                run_one,
                args.base_url,
                args.model,
                prompt,
                args.max_tokens,
                args.temperature,
                args.timeout,
                f"concurrent-r{r}-i{i}",
                args.speculative_n_max,
            )
            for r, i, prompt in concurrent_jobs
        ]
        for fut in concurrent.futures.as_completed(futs):
            concurrent_results.append(fut.result())
    concurrent_wall = time.perf_counter() - t0

    report = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "speculative_n_max": args.speculative_n_max,
        "health": health,
        "slots_before": slots,
        "single": summarize("single", single_results, single_wall),
        "concurrent": summarize("concurrent", concurrent_results, concurrent_wall),
        "single_results": [as_dict(r) for r in single_results],
        "concurrent_results": [as_dict(r) for r in sorted(concurrent_results, key=lambda x: x.label)],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "output": str(output),
        "single": report["single"],
        "concurrent": report["concurrent"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
