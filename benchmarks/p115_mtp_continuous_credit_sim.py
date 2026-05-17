#!/usr/bin/env python3
"""P115: simulate upper-bound continuous-credit from P107 MTP shadow traces.

P111 budgets a one-token MTP sidecar from aggregate accept rates. P114 shows
whether accepted events are contiguous. This script combines both ideas: given
base-state P107 shadow flags, simulate an ideal speculative loop that can cash
in up to N consecutive covered draft tokens per verifier iteration.

Important: this is still an upper bound. P107 observes every position under
base states; a real recursive MTP rollout must reproduce those draft choices
after accepting its own tokens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _event_covered(event: dict[str, Any], k: int) -> bool:
    if k <= 1:
        return bool(event.get("accepted"))
    topk = event.get("draft_topk") or event.get("draft_top2") or {}
    ids = [int(x) for x in (topk.get("ids") or [])[:k]]
    return int(event["base_next_id"]) in set(ids)


def _row_flags(row: dict[str, Any], *, k: int, skip_forced_prefix_events: bool) -> list[bool]:
    trace = list(row.get("mtp_shadow", {}).get("trace", []))
    if skip_forced_prefix_events:
        forced_len = len((row.get("forced_prefix_report") or {}).get("ids") or [])
        trace = [event for event in trace if int(event["step"]) >= forced_len]
    return [_event_covered(event, k) for event in trace]


def _simulate_flags(flags: list[bool], *, draft_depth: int) -> dict[str, Any]:
    i = 0
    iterations = 0
    accepted_drafts = 0
    full_depth_iterations = 0
    tokens = len(flags)
    while i < tokens:
        consecutive = 0
        while (
            consecutive < draft_depth
            and i + consecutive < tokens
            and flags[i + consecutive]
        ):
            consecutive += 1
        iterations += 1
        accepted_drafts += consecutive
        if consecutive == draft_depth:
            full_depth_iterations += 1
        # A verifier iteration emits accepted drafts plus one target token when
        # the trace still has a following position. This is the standard
        # speculative decode accounting and intentionally optimistic here.
        i += min(tokens - i, consecutive + 1)

    multiplier = (tokens / iterations) if iterations else 0.0
    return {
        "events": tokens,
        "iterations": iterations,
        "accepted_drafts": accepted_drafts,
        "accepted_drafts_per_iteration": (
            accepted_drafts / iterations if iterations else None
        ),
        "full_depth_iterations": full_depth_iterations,
        "zero_overhead_multiplier": multiplier,
    }


def _projection(
    sim: dict[str, Any],
    *,
    base_tps: float,
    draft_ms: float | None,
    draft_depth: int,
) -> dict[str, Any]:
    iterations = int(sim["iterations"])
    events = int(sim["events"])
    base_s = 1.0 / base_tps
    multiplier = float(sim["zero_overhead_multiplier"])
    out: dict[str, Any] = {
        "zero_overhead_tps": base_tps * multiplier,
        "zero_overhead_multiplier": multiplier,
    }
    if iterations:
        max_total_s = events / 155.0
        max_draft_s = (max_total_s / iterations) - base_s
        out["max_draft_ms_per_iteration_for_155"] = (
            max_draft_s * 1000.0 if max_draft_s >= 0.0 else None
        )
    if draft_ms is not None and iterations:
        optimistic_s = iterations * (base_s + draft_ms / 1000.0)
        serial_s = iterations * (base_s + (draft_ms * draft_depth) / 1000.0)
        out.update(
            {
                "draft_ms": draft_ms,
                "optimistic_same_draft_cost_tps": events / optimistic_s,
                "serial_per_draft_token_tps": events / serial_s,
            }
        )
    return out


def _analyze(
    report: dict[str, Any],
    *,
    skip_forced_prefix_events: bool,
    base_tps: float,
    draft_ms: float | None,
) -> dict[str, Any]:
    rows = list(report.get("rows", []))
    out: dict[str, Any] = {}
    for k in (1, 2, 4, 8):
        flags: list[bool] = []
        per_prompt: list[dict[str, Any]] = []
        for row in rows:
            row_flags = _row_flags(row, k=k, skip_forced_prefix_events=skip_forced_prefix_events)
            flags.extend(row_flags)
            if row_flags:
                row_sim = _simulate_flags(row_flags, draft_depth=1)
                per_prompt.append(
                    {
                        "id": row.get("id"),
                        "events": len(row_flags),
                        "covered": sum(1 for flag in row_flags if flag),
                        "depth1_iterations": row_sim["iterations"],
                        "depth1_multiplier": row_sim["zero_overhead_multiplier"],
                    }
                )
        by_depth: dict[str, Any] = {}
        for draft_depth in (1, 2, 4, 8):
            sim = _simulate_flags(flags, draft_depth=draft_depth)
            sim["projection"] = _projection(
                sim,
                base_tps=base_tps,
                draft_ms=draft_ms,
                draft_depth=draft_depth,
            )
            by_depth[f"depth{draft_depth}"] = sim
        out[f"top{k}"] = {
            "events": len(flags),
            "covered": sum(1 for flag in flags if flag),
            "coverage": (sum(1 for flag in flags if flag) / len(flags)) if flags else None,
            "by_depth": by_depth,
            "per_prompt_depth1": per_prompt,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p107-report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-forced-prefix-events", action="store_true")
    ap.add_argument("--production-baseline-tps", type=float, default=100.0)
    ap.add_argument(
        "--draft-ms",
        type=float,
        default=None,
        help="One-token draft cost in ms. Defaults to 1000 / summary.draft_tps when present.",
    )
    args = ap.parse_args()

    report = json.loads(Path(args.p107_report).read_text(encoding="utf-8"))
    summary = report.get("summary") or {}
    draft_ms = args.draft_ms
    if draft_ms is None and summary.get("draft_tps"):
        draft_ms = 1000.0 / float(summary["draft_tps"])

    analysis = _analyze(
        report,
        skip_forced_prefix_events=args.skip_forced_prefix_events,
        base_tps=args.production_baseline_tps,
        draft_ms=draft_ms,
    )
    top1_depth1_tps = analysis["top1"]["by_depth"]["depth1"]["projection"]["zero_overhead_tps"]
    top1_depth2_tps = analysis["top1"]["by_depth"]["depth2"]["projection"]["zero_overhead_tps"]
    top8_depth2_tps = analysis["top8"]["by_depth"]["depth2"]["projection"]["zero_overhead_tps"]
    result = {
        "schema_version": "lynn-p115-mtp-continuous-credit-sim-v1",
        "source_report": args.p107_report,
        "skip_forced_prefix_events": args.skip_forced_prefix_events,
        "production_baseline_tps": args.production_baseline_tps,
        "draft_ms": draft_ms,
        "decision": (
            "GREEN-CREDIT-SIM: top1 depth2 zero-overhead can clear 155 TPS; build a runtime prototype."
            if top1_depth2_tps >= 155.0
            else "AMBER-CREDIT-SIM: top1 needs overlap, higher accept, or top-k verification to clear 155 TPS."
        ),
        "note": "Upper-bound simulation from base-state P107 trace flags; not a recursive MTP rollout measurement.",
        "summary": {
            "top1_depth1_zero_overhead_tps": top1_depth1_tps,
            "top1_depth2_zero_overhead_tps": top1_depth2_tps,
            "top8_depth2_zero_overhead_tps": top8_depth2_tps,
        },
        "analysis": analysis,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
