#!/usr/bin/env python3
"""One-run R6000 MoE budget ladder.

The older P51 ladder reloads the resident model for every candidate. This
script keeps one runner alive, then toggles decode-time MoE budget env knobs:

* top-k limit for active routed experts;
* shared expert skip.

It is not a promotion gate. It measures how much speed is available and how
quickly greedy token IDs diverge from the safe baseline.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p37_moe_config_generate_gate import BASE_ENV, PROMPTS  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


def _set_env(updates: dict[str, str | None]) -> dict[str, str | None]:
    old = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _candidate_updates(topk_limit: int | None, skip_shared: bool) -> dict[str, str | None]:
    return {
        "LYNN_MOE_TOPK_LIMIT": None if topk_limit is None else str(topk_limit),
        "LYNN_MOE_PROFILE_TOPK_LIMIT": None if topk_limit is None else str(topk_limit),
        "LYNN_MOE_TOPK_RENORMALIZE": "1",
        "LYNN_MOE_SKIP_SHARED": "1" if skip_shared else "0",
        "LYNN_MOE_PROFILE_SKIP_SHARED": "1" if skip_shared else "0",
    }


def _prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if int(x) != int(y):
            break
        n += 1
    return n


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(row["decode_tps"]) for row in rows if row.get("decode_tps") is not None]
    prefixes = [int(row["prefix_match"]) for row in rows if row.get("prefix_match") is not None]
    return {
        "count": len(rows),
        "exact": sum(1 for row in rows if row.get("exact_match")),
        "min_prefix": min(prefixes) if prefixes else None,
        "mean_prefix": statistics.fmean(prefixes) if prefixes else None,
        "decode_tps_mean": statistics.fmean(vals) if vals else None,
        "decode_tps_median": statistics.median(vals) if vals else None,
        "decode_tps_min": min(vals) if vals else None,
        "decode_tps_max": max(vals) if vals else None,
    }


def _run_case(
    runner: LynnIncrementalRunner,
    *,
    label: str,
    topk_limit: int | None,
    skip_shared: bool,
    prompts: list[str],
    max_new: int,
    baseline_ids: list[list[int]] | None,
) -> dict[str, Any]:
    old = _set_env(_candidate_updates(topk_limit, skip_shared))
    try:
        rows: list[dict[str, Any]] = []
        for idx, prompt in enumerate(prompts):
            out = runner.generate(prompt, max_new=max_new, use_chat_template=False)
            new_ids = [int(x) for x in out["new_ids"]]
            base = None if baseline_ids is None else baseline_ids[idx]
            rows.append(
                {
                    "prompt_id": f"prompt_{idx:03d}",
                    "prompt": prompt,
                    "new_ids": new_ids,
                    "new_ids_head": new_ids[:24],
                    "completion_text": out["completion_text"][:240],
                    "decode_tps": out["timings"].get("decode_tps"),
                    "stopped_reason": out["stopped_reason"],
                    "exact_match": None if base is None else new_ids == base,
                    "prefix_match": None if base is None else _prefix_len(new_ids, base),
                }
            )
        return {
            "label": label,
            "topk_limit": topk_limit,
            "skip_shared": skip_shared,
            "summary": _summary(rows),
            "rows": rows,
        }
    finally:
        _restore_env(old)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--prompts-json")
    args = ap.parse_args()

    prompts = PROMPTS
    if args.prompts_json:
        raw = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
        prompts = [str(item["prompt"] if isinstance(item, dict) else item) for item in raw]

    env = dict(BASE_ENV)
    env["LYNN_MOE_FAST_FIXED"] = "1"
    env["LYNN_NATIVE_DOWN_BACKEND"] = "triton"
    old = _set_env(env)
    try:
        runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
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
        cases = []
        baseline_ids: list[list[int]] | None = None
        for label, topk_limit, skip_shared in configs:
            case = _run_case(
                runner,
                label=label,
                topk_limit=topk_limit,
                skip_shared=skip_shared,
                prompts=prompts,
                max_new=args.max_new,
                baseline_ids=baseline_ids,
            )
            if baseline_ids is None:
                baseline_ids = [
                    [int(x) for x in row["new_ids"]]
                    for row in case["rows"]
                ]
                # Rerun exact baseline metadata against itself for summary
                for row in case["rows"]:
                    row["exact_match"] = True
                    row["prefix_match"] = len(row["new_ids"])
                case["summary"] = _summary(case["rows"])
            cases.append(case)

        baseline_tps = cases[0]["summary"]["decode_tps_median"]
        for case in cases:
            tps = case["summary"]["decode_tps_median"]
            case["median_speedup_vs_baseline"] = (tps / baseline_tps) if tps and baseline_tps else None
        report = {
            "schema_version": "lynn-r6000-moe-budget-one-runner-v1",
            "model": args.model,
            "max_new": args.max_new,
            "prompt_count": len(prompts),
            "cases": cases,
            "notes": [
                "One runner is reused; env knobs are decode-time MoE budget switches.",
                "This is a speed/quality frontier probe, not a promotion gate.",
            ],
        }
    finally:
        _restore_env(old)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([
        {
            "label": case["label"],
            "median_tps": case["summary"]["decode_tps_median"],
            "speedup": case["median_speedup_vs_baseline"],
            "exact": case["summary"]["exact"],
            "min_prefix": case["summary"]["min_prefix"],
        }
        for case in report["cases"]
    ], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
