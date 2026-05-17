#!/usr/bin/env python3
"""OpenAI-compatible JSON guard TPS gate.

This gate intentionally measures the service-facing path, not an isolated
kernel. It sends repeated `chat/completions` requests with
`response_format={"type":"json_object"}`, validates parseability and stop
semantics, then compares decode TPS to a target.

Use it for R6000 Lynn server, Spark Lynn server, or a llama.cpp OpenAI server
when the same prompts/model/token budget are exposed behind an OpenAI-compatible
endpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.request
from typing import Any


DEFAULT_PROMPTS = [
    "Return one JSON object with keys city and unit for Tokyo in celsius. No markdown.",
    "Return one JSON object with keys city and unit for Paris in fahrenheit. No markdown.",
    "Return one JSON object with key status and value ok. No markdown.",
    "Return one JSON object with keys tool and arguments for get_weather city Tokyo unit celsius. No markdown.",
]


def _load_prompts(path: str | None, inline: list[str]) -> list[str]:
    if not path:
        return inline
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts: list[str] = []
    for item in raw:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict):
            prompts.append(str(item["prompt"]))
        else:
            raise TypeError(f"prompt must be string or object, got {type(item)}")
    return prompts


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="Server root, e.g. http://127.0.0.1:18156")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts-file")
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--target-decode-tps", type=float, default=155.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--system", default="You output JSON only.")
    args = ap.parse_args()

    prompts = _load_prompts(args.prompts_file, args.prompts)
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    rows: list[dict[str, Any]] = []
    for idx in range(args.requests):
        prompt = prompts[idx % len(prompts)]
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": args.system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "response_format": {"type": "json_object"},
        }
        t0 = time.time()
        parsed = _post_json(url, payload, args.timeout)
        elapsed = time.time() - t0
        choice = parsed["choices"][0]
        text = choice["message"]["content"]
        metrics = parsed.get("_lynn_engine_metrics", {})
        try:
            json.loads(text)
            json_parse = True
            json_error = None
        except Exception as exc:  # noqa: BLE001 - diagnostic gate
            json_parse = False
            json_error = str(exc)
        rows.append(
            {
                "idx": idx,
                "prompt": prompt,
                "elapsed_seconds": elapsed,
                "finish_reason": choice.get("finish_reason"),
                "stopped_reason": metrics.get("stopped_reason"),
                "completion_tokens": parsed.get("usage", {}).get("completion_tokens"),
                "decode_tps": metrics.get("timings", {}).get("decode_tps"),
                "tokens_per_second_http": metrics.get("tokens_per_second"),
                "json_parse": json_parse,
                "json_error": json_error,
                "format_ok": text.lstrip().startswith("{") and "```" not in text,
                "text": text,
                "forced_raw_all_matched": (
                    metrics.get("format_guard", {})
                    .get("forced_prefix", {})
                    .get("raw_all_matched")
                ),
            }
        )

    decode = [float(row["decode_tps"]) for row in rows if row.get("decode_tps") is not None]
    http_tps = [
        float(row["tokens_per_second_http"])
        for row in rows
        if row.get("tokens_per_second_http") is not None
    ]
    summary = {
        "request_count": len(rows),
        "all_json_parse": all(row["json_parse"] for row in rows),
        "all_format_ok": all(row["format_ok"] for row in rows),
        "all_finish_stop": all(row["finish_reason"] == "stop" for row in rows),
        "decode_tps_mean": statistics.fmean(decode) if decode else None,
        "decode_tps_min": min(decode) if decode else None,
        "decode_tps_max": max(decode) if decode else None,
        "http_tps_mean": statistics.fmean(http_tps) if http_tps else None,
    }
    target_met = (
        bool(summary["all_json_parse"])
        and bool(summary["all_format_ok"])
        and bool(summary["all_finish_stop"])
        and summary["decode_tps_mean"] is not None
        and float(summary["decode_tps_mean"]) >= args.target_decode_tps
    )
    report = {
        "schema_version": "lynn-openai-json-guard-tps-gate-v1",
        "decision": (
            f"GREEN: JSON guard serving meets target {args.target_decode_tps:.2f} decode TPS."
            if target_met
            else f"RED: JSON guard serving is below target {args.target_decode_tps:.2f} decode TPS or failed format checks."
        ),
        "base_url": args.base_url,
        "model": args.model,
        "target_decode_tps": args.target_decode_tps,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "summary": summary,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if target_met else 1


if __name__ == "__main__":
    raise SystemExit(main())
