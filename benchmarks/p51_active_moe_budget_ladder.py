#!/usr/bin/env python3
"""P51: active-MoE budget ladder.

This is intentionally not an exact-greedy promotion gate.  P48-P50 proved that
tiny accumulation drift can flip tokens, so P51 asks a product question:

    If we spend fewer active experts or drop the shared expert, how much speed
    is available and how ugly does generation get?

The default engine remains exact.  P51 candidates are opt-in quality-gated
research profiles for the 155 TPS line.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from statistics import mean, median
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p37_moe_config_generate_gate import BASE_ENV, PROMPTS  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


def _set_env(updates: dict[str, str]) -> dict[str, str | None]:
    old = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(r["timings"]["decode_tps"]) for r in rows if r["timings"].get("decode_tps")]
    return {
        "count": len(rows),
        "decode_tps_mean": mean(vals) if vals else None,
        "decode_tps_median": median(vals) if vals else None,
        "decode_tps_min": min(vals) if vals else None,
        "decode_tps_max": max(vals) if vals else None,
        "stop_reasons": {reason: sum(1 for r in rows if r["stopped_reason"] == reason) for reason in sorted({r["stopped_reason"] for r in rows})},
    }


def _candidate_env(*, topk_limit: int | None, skip_shared: bool) -> dict[str, str]:
    env = dict(BASE_ENV)
    env["LYNN_MOE_FAST_FIXED"] = "1"
    env["LYNN_NATIVE_DOWN_BACKEND"] = "triton"
    env["LYNN_MOE_TOPK_RENORMALIZE"] = "1"
    if topk_limit is None:
        env.pop("LYNN_MOE_TOPK_LIMIT", None)
        env.pop("LYNN_MOE_PROFILE_TOPK_LIMIT", None)
    else:
        env["LYNN_MOE_TOPK_LIMIT"] = str(topk_limit)
    env["LYNN_MOE_SKIP_SHARED"] = "1" if skip_shared else "0"
    return env


def _run_case(
    model: str,
    *,
    label: str,
    topk_limit: int | None,
    skip_shared: bool,
    prompts: list[str],
    max_new: int,
) -> dict[str, Any]:
    env = _candidate_env(topk_limit=topk_limit, skip_shared=skip_shared)
    old = _set_env(env)
    try:
        runner = LynnIncrementalRunner(model, device="cuda", dtype=torch.bfloat16, verbose=False)
        rows = []
        for idx, prompt in enumerate(prompts):
            out = runner.generate(prompt, max_new=max_new, use_chat_template=False)
            rows.append({
                "prompt_id": f"prompt_{idx:03d}",
                "prompt": prompt,
                "new_ids": out["new_ids"],
                "completion_text": out["completion_text"],
                "completion_text_raw": out["completion_text_raw"],
                "stopped_reason": out["stopped_reason"],
                "timings": out["timings"],
            })
        return {
            "label": label,
            "topk_limit": topk_limit,
            "skip_shared": skip_shared,
            "summary": _summary(rows),
            "samples": [
                {
                    "prompt_id": row["prompt_id"],
                    "new_ids_head": row["new_ids"][:16],
                    "completion_text": row["completion_text"][:240],
                    "stopped_reason": row["stopped_reason"],
                    "decode_tps": row["timings"]["decode_tps"],
                }
                for row in rows
            ],
        }
    finally:
        _restore_env(old)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--prompts-jsonl")
    args = ap.parse_args()

    prompts = PROMPTS
    if args.prompts_jsonl:
        prompts = []
        with open(args.prompts_jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompt = item.get("prompt") or item.get("input") or item.get("question") or item.get("problem")
                if prompt:
                    prompts.append(str(prompt))
        if not prompts:
            raise ValueError(f"No prompts found in {args.prompts_jsonl}")

    configs = [
        ("baseline_top8_shared", None, False),
        ("top6_shared", 6, False),
        ("top4_shared", 4, False),
        ("top2_shared", 2, False),
        ("top1_shared", 1, False),
        ("top8_skip_shared", None, True),
        ("top6_skip_shared", 6, True),
        ("top4_skip_shared", 4, True),
        ("top2_skip_shared", 2, True),
        ("top1_skip_shared", 1, True),
    ]
    cases = [
        _run_case(
            args.model,
            label=label,
            topk_limit=topk_limit,
            skip_shared=skip_shared,
            prompts=prompts,
            max_new=args.max_new,
        )
        for label, topk_limit, skip_shared in configs
    ]
    baseline = cases[0]["summary"]["decode_tps_median"]
    for case in cases:
        med = case["summary"]["decode_tps_median"]
        case["median_speedup_vs_baseline"] = (med / baseline) if med and baseline else None
    result = {
        "schema_version": "lynn-engine-p51-active-moe-budget-ladder-v1",
        "model": args.model,
        "max_new": args.max_new,
        "prompt_count": len(prompts),
        "cases": cases,
        "notes": [
            "This is not an exact-greedy promotion gate.",
            "Candidates are opt-in quality-gated profiles for speed/quality tradeoff exploration.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([
        {
            "label": c["label"],
            "median_tps": c["summary"]["decode_tps_median"],
            "speedup": c["median_speedup_vs_baseline"],
            "stop_reasons": c["summary"]["stop_reasons"],
        }
        for c in cases
    ], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
