#!/usr/bin/env python3
"""OpenAI-compatible structured serving TPS gate.

This is a service-facing gate for the Qwen3.6 W4A16 fast path. It checks that
JSON/tool-call/code/YAML/short-answer prompts keep their expected shape while
also recording Lynn server decode TPS metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.request
from typing import Any


DEFAULT_SPECS: list[dict[str, Any]] = [
    {
        "id": "json_city_metric",
        "prompt": "Return one JSON object with keys city and unit for Tokyo in celsius. No markdown.",
        "system": "You output JSON only.",
        "response_format": "json_object",
        "starts_with": "{",
        "parse_json": True,
        "must_contain": ["city", "unit"],
        "forbid": ["```"],
    },
    {
        "id": "tool_args_weather",
        "prompt": "Return one JSON object with keys tool and arguments for get_weather city Tokyo unit celsius. No markdown.",
        "system": "You output function-call arguments as plain JSON only.",
        "response_format": "json_object",
        "starts_with": "{",
        "parse_json": True,
        "must_contain": ["tool", "arguments", "Tokyo"],
        "forbid": ["```"],
    },
    {
        "id": "python_slugify",
        "prompt": "Only output Python code, no markdown. Define slugify(text: str) -> str that lowercases and replaces spaces with hyphens.",
        "system": "You output code only.",
        "starts_with_any": ["def slugify", "import re\n\ndef slugify"],
        "must_contain": ["def slugify", "return"],
        "forbid": ["```"],
    },
    {
        "id": "yaml_request_body",
        "prompt": "Output only an OpenAPI YAML requestBody for JSON object {name: string}. No explanation.",
        "system": "You output YAML only. A yaml code fence is acceptable if the content is plain YAML.",
        "starts_with_any": ["requestBody:", "content:", "```yaml"],
        "must_contain": ["application/json", "name", "string"],
    },
    {
        "id": "router_zh",
        "prompt": "用一句中文短句说明 MoE router 的作用。必须以 router 开头,必须包含 动态分配、专家。",
        "system": "严格遵守用户给定的格式要求。",
        "starts_with": "router",
        "must_contain": ["动态分配", "专家"],
    },
    {
        "id": "linear_attention_zh",
        "prompt": "用一句中文短句说明 linear attention 适合长上下文的原因。必须包含 计算复杂度、线性。",
        "system": "严格遵守用户给定的格式要求。",
        "must_contain": ["计算复杂度", "线性"],
    },
    {
        "id": "short_math",
        "prompt": "Answer only with the number: 45 miles per hour for 3 hours is how many miles?",
        "system": "Answer with the final value only.",
        "starts_with": "135",
        "forbid": ["\n"],
    },
]


def _load_specs(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return DEFAULT_SPECS
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("prompt specs file must contain a JSON list")
    return [dict(item) for item in raw]


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _format_checks(text: str, spec: dict[str, Any]) -> dict[str, Any]:
    stripped = text.lstrip()
    starts_with = spec.get("starts_with")
    starts_any = list(spec.get("starts_with_any") or [])
    if starts_with is not None:
        starts_ok = stripped.startswith(str(starts_with))
    elif starts_any:
        starts_ok = any(stripped.startswith(str(item)) for item in starts_any)
    else:
        starts_ok = True

    must_contain = [str(item) for item in spec.get("must_contain") or []]
    missing = [item for item in must_contain if item not in text]
    forbid = [str(item) for item in spec.get("forbid") or []]
    forbidden_hits = [item for item in forbid if item in text]

    parse_json = bool(spec.get("parse_json"))
    json_ok = True
    json_error = None
    parsed_keys: list[str] | None = None
    if parse_json:
        try:
            parsed = json.loads(text)
            json_ok = isinstance(parsed, dict)
            if isinstance(parsed, dict):
                parsed_keys = sorted(str(key) for key in parsed.keys())
        except Exception as exc:  # noqa: BLE001 - diagnostic gate
            json_ok = False
            json_error = str(exc)

    return {
        "starts_ok": starts_ok,
        "missing": missing,
        "forbidden_hits": forbidden_hits,
        "json_ok": json_ok,
        "json_error": json_error,
        "parsed_keys": parsed_keys,
        "ok": starts_ok and not missing and not forbidden_hits and json_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="Server root, e.g. http://127.0.0.1:18166")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-specs-file")
    parser.add_argument("--requests", type=int, default=14)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--target-decode-tps", type=float, default=75.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()

    specs = _load_specs(args.prompt_specs_file)
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    rows: list[dict[str, Any]] = []

    for idx in range(args.requests):
        spec = specs[idx % len(specs)]
        payload: dict[str, Any] = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": str(spec.get("system", "Follow the requested output format exactly."))},
                {"role": "user", "content": str(spec["prompt"])},
            ],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        if spec.get("response_format") == "json_object":
            payload["response_format"] = {"type": "json_object"}

        t0 = time.time()
        response = _post_json(url, payload, args.timeout)
        elapsed = time.time() - t0
        choice = response["choices"][0]
        text = choice["message"]["content"]
        metrics = response.get("_lynn_engine_metrics", {})
        rows.append(
            {
                "idx": idx,
                "id": spec.get("id", f"spec_{idx % len(specs)}"),
                "prompt": spec["prompt"],
                "elapsed_seconds": elapsed,
                "finish_reason": choice.get("finish_reason"),
                "stopped_reason": metrics.get("stopped_reason"),
                "completion_tokens": response.get("usage", {}).get("completion_tokens"),
                "decode_tps": metrics.get("timings", {}).get("decode_tps"),
                "tokens_per_second_http": metrics.get("tokens_per_second"),
                "format": _format_checks(text, spec),
                "text": text,
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
        "format_ok": sum(1 for row in rows if row["format"]["ok"]),
        "all_format_ok": all(row["format"]["ok"] for row in rows),
        "all_finish_stop": all(row["finish_reason"] == "stop" for row in rows),
        "decode_tps_mean": statistics.fmean(decode) if decode else None,
        "decode_tps_min": min(decode) if decode else None,
        "decode_tps_max": max(decode) if decode else None,
        "http_tps_mean": statistics.fmean(http_tps) if http_tps else None,
    }
    target_met = (
        bool(summary["all_format_ok"])
        and bool(summary["all_finish_stop"])
        and summary["decode_tps_mean"] is not None
        and float(summary["decode_tps_mean"]) >= args.target_decode_tps
    )
    report = {
        "schema_version": "lynn-openai-structured-tps-gate-v1",
        "decision": (
            f"GREEN: structured serving meets target {args.target_decode_tps:.2f} decode TPS."
            if target_met
            else f"RED: structured serving is below target {args.target_decode_tps:.2f} decode TPS or failed format checks."
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
