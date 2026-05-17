#!/usr/bin/env python3
"""P114: analyze accepted-token streaks in P107 MTP shadow traces.

This is an upper-bound diagnostic for multi-token or multi-candidate runtime
work. P107 records whether the MTP draft top-k contains the base greedy token at
each generated position. Consecutive covered positions are the only places where
multi-token credit could matter; isolated hits are already captured by one-token
speculation.

Important: these are trace streaks under base states, not a real recursive MTP
rollout. Treat them as a prioritization signal for runtime investment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


def _event_covered(event: dict[str, Any], k: int) -> bool:
    if k <= 1:
        return bool(event.get("accepted"))
    topk = event.get("draft_topk") or event.get("draft_top2") or {}
    ids = [int(x) for x in (topk.get("ids") or [])[:k]]
    return int(event["base_next_id"]) in set(ids)


def _run_lengths(flags: list[bool]) -> list[int]:
    runs: list[int] = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def _stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "run_count": 0,
            "max_run": 0,
            "mean_run": 0.0,
            "median_run": 0.0,
            "runs_ge2": 0,
            "runs_ge3": 0,
            "covered_tokens_in_runs_ge2": 0,
            "covered_tokens_in_runs_ge3": 0,
        }
    return {
        "run_count": len(values),
        "max_run": max(values),
        "mean_run": statistics.fmean(values),
        "median_run": statistics.median(values),
        "runs_ge2": sum(1 for x in values if x >= 2),
        "runs_ge3": sum(1 for x in values if x >= 3),
        "covered_tokens_in_runs_ge2": sum(x for x in values if x >= 2),
        "covered_tokens_in_runs_ge3": sum(x for x in values if x >= 3),
    }


def _collect_trace(report: dict[str, Any], *, skip_forced_prefix_events: bool) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for row in report.get("rows", []):
        row_trace = list(row.get("mtp_shadow", {}).get("trace", []))
        if skip_forced_prefix_events:
            forced_len = len((row.get("forced_prefix_report") or {}).get("ids") or [])
            row_trace = [event for event in row_trace if int(event["step"]) >= forced_len]
        for event in row_trace:
            event = dict(event)
            event["prompt_id"] = row.get("id")
            trace.append(event)
    return trace


def _analyze(report: dict[str, Any], *, skip_forced_prefix_events: bool) -> dict[str, Any]:
    rows = list(report.get("rows", []))
    per_k: dict[str, Any] = {}
    for k in (1, 2, 4, 8):
        all_flags: list[bool] = []
        per_prompt: list[dict[str, Any]] = []
        for row in rows:
            row_trace = list(row.get("mtp_shadow", {}).get("trace", []))
            if skip_forced_prefix_events:
                forced_len = len((row.get("forced_prefix_report") or {}).get("ids") or [])
                row_trace = [event for event in row_trace if int(event["step"]) >= forced_len]
            flags = [_event_covered(event, k) for event in row_trace]
            runs = _run_lengths(flags)
            all_flags.extend(flags)
            per_prompt.append(
                {
                    "id": row.get("id"),
                    "events": len(flags),
                    "covered": sum(1 for flag in flags if flag),
                    "coverage": (sum(1 for flag in flags if flag) / len(flags)) if flags else None,
                    "runs": runs,
                    "max_run": max(runs) if runs else 0,
                }
            )
        runs = _run_lengths(all_flags)
        covered = sum(1 for flag in all_flags if flag)
        stat = _stats(runs)
        stat.update(
            {
                "events": len(all_flags),
                "covered": covered,
                "coverage": covered / len(all_flags) if all_flags else None,
                "covered_token_share_in_runs_ge2": (
                    stat["covered_tokens_in_runs_ge2"] / covered if covered else None
                ),
                "covered_token_share_in_runs_ge3": (
                    stat["covered_tokens_in_runs_ge3"] / covered if covered else None
                ),
                "per_prompt": per_prompt,
            }
        )
        per_k[f"top{k}"] = stat
    return per_k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p107-report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-forced-prefix-events", action="store_true")
    args = ap.parse_args()

    report = json.loads(Path(args.p107_report).read_text(encoding="utf-8"))
    analysis = _analyze(report, skip_forced_prefix_events=args.skip_forced_prefix_events)
    top1_max = analysis["top1"]["max_run"]
    top8_max = analysis["top8"]["max_run"]
    result = {
        "schema_version": "lynn-p114-mtp-shadow-streak-analysis-v1",
        "source_report": args.p107_report,
        "skip_forced_prefix_events": args.skip_forced_prefix_events,
        "decision": (
            "GREEN-STREAK: trace has top1 runs >=3; multi-token credit is worth a runtime prototype."
            if top1_max >= 3
            else "AMBER-STREAK: top1 streaks are short; multi-token work likely needs top-k/rerank or overlap."
        ),
        "note": "Streaks are P107 base-state shadow upper bounds, not recursive MTP rollout measurements.",
        "analysis": analysis,
        "summary": {
            "top1_max_run": top1_max,
            "top1_runs_ge2": analysis["top1"]["runs_ge2"],
            "top1_covered_token_share_in_runs_ge2": analysis["top1"]["covered_token_share_in_runs_ge2"],
            "top8_max_run": top8_max,
            "top8_runs_ge2": analysis["top8"]["runs_ge2"],
            "top8_covered_token_share_in_runs_ge2": analysis["top8"]["covered_token_share_in_runs_ge2"],
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
