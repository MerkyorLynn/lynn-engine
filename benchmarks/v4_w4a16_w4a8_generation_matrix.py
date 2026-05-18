#!/usr/bin/env python3
"""V4 35B W4A16 vs W4A8 generation matrix.

This benchmark runs one Lynn-native weight-only NVFP4 artifact in three decode
activation modes:

- off: W4A16, BF16 activations
- gateup: W4A8-like FP8 fake quant on MoE input/gate-up activation
- full: W4A8-like FP8 fake quant on MoE input/gate-up and intermediate/down

It is intentionally a generation gate, not an MMLU substitute. The goal is to
decide whether W4A8 is worth promoting from a speed bridge after W4A16 quality
is measured.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


DEFAULT_PROMPTS = [
    {
        "id": "json_city_metric",
        "prompt": "Return one JSON object with keys city and unit for Tokyo in metric units. No markdown.",
    },
    {
        "id": "python_slugify",
        "prompt": "Only output Python code, no markdown. Define slugify(text: str) -> str that lowercases and replaces spaces with hyphens.",
    },
    {
        "id": "moe_router_zh",
        "prompt": "用一句中文短句说明 MoE router 的作用。必须以 router 开头,必须包含 动态分配、专家。",
    },
    {
        "id": "linear_attention_zh",
        "prompt": "用一句中文短句说明 linear attention 适合长上下文的原因。必须包含 计算复杂度、线性。",
    },
    {
        "id": "yaml_request_body",
        "prompt": "Output only an OpenAPI YAML requestBody for JSON object {name: string}. No explanation.",
    },
    {
        "id": "short_math",
        "prompt": "Answer only with the number: 45 miles per hour for 3 hours is how many miles?",
    },
]


def _load_prompts(path: str | None) -> list[dict[str, str]]:
    if not path:
        return DEFAULT_PROMPTS
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, str):
            prompts.append({"id": f"prompt_{idx:03d}", "prompt": item})
        elif isinstance(item, dict):
            prompts.append(
                {
                    "id": str(item.get("id", f"prompt_{idx:03d}")),
                    "prompt": str(item["prompt"]),
                }
            )
        else:
            raise TypeError(f"prompt must be string or object, got {type(item)}")
    return prompts


def _same_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if int(x) != int(y):
            break
        n += 1
    return n


def _run_mode(
    runner: LynnIncrementalRunner,
    prompts: list[dict[str, str]],
    *,
    mode: str,
    max_new: int,
    top_k: int,
    use_chat_template: bool,
) -> dict[str, Any]:
    os.environ["LYNN_W4A8_FAKE_QUANT_ACTIVE"] = mode
    rows = []
    for item in prompts:
        out = runner.generate(
            item["prompt"],
            max_new=max_new,
            top_k=top_k,
            use_chat_template=use_chat_template,
        )
        rows.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "new_ids": out["new_ids"],
                "completion_text": out["completion_text"],
                "completion_text_raw": out["completion_text_raw"],
                "stopped_reason": out["stopped_reason"],
                "decode_tps": out["timings"].get("decode_tps"),
                "decode_seconds": out["timings"].get("decode_seconds"),
                "topk_trace": out.get("topk_trace", []),
            }
        )
    tps = [float(row["decode_tps"]) for row in rows if row.get("decode_tps") is not None]
    return {
        "mode": mode,
        "rows": rows,
        "mean_decode_tps": statistics.fmean(tps) if tps else None,
        "min_decode_tps": min(tps) if tps else None,
    }


def _compare(reference: dict[str, Any], candidate: dict[str, Any], *, max_new: int) -> dict[str, Any]:
    rows = []
    for ref, cand in zip(reference["rows"], candidate["rows"]):
        prefix = _same_prefix(ref["new_ids"], cand["new_ids"])
        rows.append(
            {
                "id": ref["id"],
                "prompt": ref["prompt"],
                "exact": ref["new_ids"] == cand["new_ids"],
                "same_prefix_tokens": prefix,
                "first_diff_index": None if prefix == max_new else prefix,
                "reference_text": ref["completion_text"],
                "candidate_text": cand["completion_text"],
            }
        )
    prefixes = [int(row["same_prefix_tokens"]) for row in rows]
    exact = sum(1 for row in rows if row["exact"])
    return {
        "reference": reference["mode"],
        "candidate": candidate["mode"],
        "exact": exact,
        "prompt_count": len(rows),
        "exact_rate": exact / len(rows) if rows else 0.0,
        "min_same_prefix_tokens": min(prefixes) if prefixes else None,
        "mean_same_prefix_tokens": statistics.fmean(prefixes) if prefixes else None,
        "rows": rows,
    }


def _decision(compare_rows: list[dict[str, Any]], *, max_new: int) -> str:
    if not compare_rows:
        return "RED: no comparisons produced."
    full = next((row for row in compare_rows if row["candidate"] == "full"), compare_rows[-1])
    exact = int(full["exact"])
    total = int(full["prompt_count"])
    min_prefix = full["min_same_prefix_tokens"]
    if total and exact == total:
        return "GREEN: W4A8-full matches W4A16 on this generation matrix."
    if min_prefix is not None and min_prefix >= max_new * 0.75:
        return "AMBER: W4A8-full diverges late; benchmark speed and consider recovery."
    return "RED: W4A8-full diverges early versus W4A16; keep W4A16 as quality route."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts-file")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=["off", "gateup", "full"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--use-chat-template", action="store_true")
    args = ap.parse_args()

    prompts = _load_prompts(args.prompts_file)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    os.environ.setdefault("LYNN_MOE_IMPL", "packed_nvfp4")
    os.environ.setdefault("LYNN_W4A8_FAKE_QUANT_FORMAT", "e4m3")
    os.environ.setdefault("LYNN_W4A8_FAKE_QUANT_GRANULARITY", "per16")
    os.environ.setdefault("LYNN_NATIVE_FP4_LM_HEAD", "1")
    os.environ.setdefault("LYNN_ROUTER_TOPK_SORTED", "1")

    runner = LynnIncrementalRunner(
        args.model,
        device=args.device,
        dtype=dtype,
        max_seq_len=4096,
        verbose=True,
    )
    try:
        mode_results = [
            _run_mode(
                runner,
                prompts,
                mode=mode,
                max_new=args.max_new,
                top_k=args.top_k,
                use_chat_template=args.use_chat_template,
            )
            for mode in args.modes
        ]
    finally:
        os.environ["LYNN_W4A8_FAKE_QUANT_ACTIVE"] = "off"
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    reference = mode_results[0]
    comparisons = [
        _compare(reference, candidate, max_new=args.max_new)
        for candidate in mode_results[1:]
    ]
    result = {
        "schema_version": "lynn-v4-w4a16-w4a8-generation-matrix-v1",
        "decision": _decision(comparisons, max_new=args.max_new),
        "model": args.model,
        "max_new": args.max_new,
        "top_k": args.top_k,
        "use_chat_template": args.use_chat_template,
        "env": {
            "LYNN_MOE_IMPL": os.environ.get("LYNN_MOE_IMPL"),
            "LYNN_W4A8_FAKE_QUANT_FORMAT": os.environ.get("LYNN_W4A8_FAKE_QUANT_FORMAT"),
            "LYNN_W4A8_FAKE_QUANT_GRANULARITY": os.environ.get("LYNN_W4A8_FAKE_QUANT_GRANULARITY"),
            "LYNN_NATIVE_FP4_LM_HEAD": os.environ.get("LYNN_NATIVE_FP4_LM_HEAD"),
        },
        "modes": mode_results,
        "comparisons_vs_reference": comparisons,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "comparisons": comparisons}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
