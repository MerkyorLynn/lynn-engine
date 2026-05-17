#!/usr/bin/env python3
"""P116: turn MTP shadow traces into route-level speculative policy targets.

P115 shows whether continuous-credit MTP could clear 155 TPS in aggregate. For
serving work we also need route policy: which prompt families deserve MTP first,
and what runtime cost cut or reranker would be required for each.

This probe reads a P107 shadow report directly. It remains an upper-bound
analysis because P107 trace positions are base-state observations, not recursive
MTP rollout measurements.
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


def _row_trace(row: dict[str, Any], *, skip_forced_prefix_events: bool) -> list[dict[str, Any]]:
    trace = list(row.get("mtp_shadow", {}).get("trace", []))
    if skip_forced_prefix_events:
        forced_len = len((row.get("forced_prefix_report") or {}).get("ids") or [])
        trace = [event for event in trace if int(event["step"]) >= forced_len]
    return trace


def _simulate(flags: list[bool], *, depth: int) -> dict[str, Any]:
    i = 0
    iterations = 0
    accepted_drafts = 0
    full_depth_iterations = 0
    while i < len(flags):
        consecutive = 0
        while consecutive < depth and i + consecutive < len(flags) and flags[i + consecutive]:
            consecutive += 1
        iterations += 1
        accepted_drafts += consecutive
        full_depth_iterations += int(consecutive == depth)
        i += min(len(flags) - i, consecutive + 1)
    return {
        "events": len(flags),
        "iterations": iterations,
        "accepted_drafts": accepted_drafts,
        "accepted_drafts_per_iteration": accepted_drafts / iterations if iterations else None,
        "full_depth_iterations": full_depth_iterations,
        "zero_overhead_multiplier": len(flags) / iterations if iterations else 0.0,
    }


def _project(
    sim: dict[str, Any],
    *,
    base_tps: float,
    draft_ms: float,
    depth: int,
    target_tps: float,
) -> dict[str, Any]:
    events = int(sim["events"])
    iterations = int(sim["iterations"])
    if not events or not iterations:
        return {}
    base_s = 1.0 / base_tps
    zero_tps = base_tps * float(sim["zero_overhead_multiplier"])
    budget_s = (events / target_tps / iterations) - base_s
    same_cost_s = base_s + draft_ms / 1000.0
    serial_s = base_s + (draft_ms * depth) / 1000.0
    budget_ms = budget_s * 1000.0 if budget_s >= 0 else None
    return {
        "zero_overhead_tps": zero_tps,
        "optimistic_same_draft_cost_tps": events / (iterations * same_cost_s),
        "serial_per_draft_token_tps": events / (iterations * serial_s),
        "max_draft_ms_per_iteration_for_target": budget_ms,
        "same_cost_meets_target": events / (iterations * same_cost_s) >= target_tps,
        "serial_per_draft_token_meets_target": events / (iterations * serial_s) >= target_tps,
        "required_same_cost_cut_ms": (
            max(0.0, draft_ms - budget_ms) if budget_ms is not None else None
        ),
        "required_same_cost_cut_fraction": (
            max(0.0, draft_ms - budget_ms) / draft_ms if budget_ms is not None and draft_ms > 0 else None
        ),
        "required_serial_cost_cut_ms": (
            max(0.0, draft_ms * depth - budget_ms) if budget_ms is not None else None
        ),
        "required_serial_cost_cut_fraction": (
            max(0.0, draft_ms * depth - budget_ms) / (draft_ms * depth)
            if budget_ms is not None and draft_ms > 0
            else None
        ),
    }


def _metrics_for_flags(
    flags: list[bool],
    *,
    base_tps: float,
    draft_ms: float,
    target_tps: float,
) -> dict[str, Any]:
    by_depth: dict[str, Any] = {}
    for depth in (1, 2, 4, 8):
        sim = _simulate(flags, depth=depth)
        by_depth[f"depth{depth}"] = {
            **sim,
            "projection": _project(
                sim,
                base_tps=base_tps,
                draft_ms=draft_ms,
                depth=depth,
                target_tps=target_tps,
            ),
        }
    return {
        "events": len(flags),
        "covered": sum(1 for flag in flags if flag),
        "coverage": sum(1 for flag in flags if flag) / len(flags) if flags else None,
        "by_depth": by_depth,
    }


def _recommend(metrics: dict[str, Any], *, target_tps: float) -> str:
    top1_d2 = metrics["top1"]["by_depth"]["depth2"]["projection"]
    top8_d2 = metrics["top8"]["by_depth"]["depth2"]["projection"]
    top8_d4 = metrics["top8"]["by_depth"]["depth4"]["projection"]
    if top1_d2.get("same_cost_meets_target"):
        return "ENABLE_TOP1_DEPTH2: current one-draft-cost iteration clears target in the upper-bound trace."
    if top1_d2.get("zero_overhead_tps", 0.0) >= target_tps:
        return "TOP1_DEPTH2_NEEDS_OVERLAP: top1 continuity is enough, but draft cost must be hidden or cut."
    if top8_d2.get("same_cost_meets_target"):
        return "TOP8_DEPTH2_RERANK_CANDIDATE: needs a verifier/reranker, but cost budget is plausible."
    if top8_d4.get("zero_overhead_tps", 0.0) >= target_tps:
        return "TOP8_DEPTH4_ONLY: top-k has theoretical credit, but runtime/rerank burden is high."
    return "DISABLE_OR_RETRAIN: current sidecar trace is not a good MTP serving route."


def _analyze_rows(
    rows: list[dict[str, Any]],
    *,
    skip_forced_prefix_events: bool,
    base_tps: float,
    draft_ms: float,
    target_tps: float,
) -> dict[str, Any]:
    out_rows: list[dict[str, Any]] = []
    aggregate_flags: dict[int, list[bool]] = {1: [], 2: [], 4: [], 8: []}
    for row in rows:
        trace = _row_trace(row, skip_forced_prefix_events=skip_forced_prefix_events)
        row_metrics: dict[str, Any] = {}
        for k in (1, 2, 4, 8):
            flags = [_event_covered(event, k) for event in trace]
            aggregate_flags[k].extend(flags)
            row_metrics[f"top{k}"] = _metrics_for_flags(
                flags,
                base_tps=base_tps,
                draft_ms=draft_ms,
                target_tps=target_tps,
            )
        out_rows.append(
            {
                "id": row.get("id"),
                "events": len(trace),
                "recommendation": _recommend(row_metrics, target_tps=target_tps),
                "metrics": row_metrics,
            }
        )
    aggregate_metrics = {
        f"top{k}": _metrics_for_flags(
            aggregate_flags[k],
            base_tps=base_tps,
            draft_ms=draft_ms,
            target_tps=target_tps,
        )
        for k in (1, 2, 4, 8)
    }
    return {
        "aggregate": {
            "recommendation": _recommend(aggregate_metrics, target_tps=target_tps),
            "metrics": aggregate_metrics,
        },
        "routes": out_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p107-report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-forced-prefix-events", action="store_true")
    ap.add_argument("--production-baseline-tps", type=float, default=100.0)
    ap.add_argument("--target-tps", type=float, default=155.0)
    ap.add_argument("--draft-ms", type=float)
    args = ap.parse_args()

    report = json.loads(Path(args.p107_report).read_text(encoding="utf-8"))
    summary = report.get("summary") or {}
    draft_ms = args.draft_ms
    if draft_ms is None:
        if not summary.get("draft_tps"):
            raise ValueError("pass --draft-ms when source report has no summary.draft_tps")
        draft_ms = 1000.0 / float(summary["draft_tps"])

    analysis = _analyze_rows(
        list(report.get("rows", [])),
        skip_forced_prefix_events=args.skip_forced_prefix_events,
        base_tps=args.production_baseline_tps,
        draft_ms=draft_ms,
        target_tps=args.target_tps,
    )
    routes_by_recommendation: dict[str, int] = {}
    for row in analysis["routes"]:
        key = row["recommendation"].split(":", 1)[0]
        routes_by_recommendation[key] = routes_by_recommendation.get(key, 0) + 1
    result = {
        "schema_version": "lynn-p116-mtp-route-policy-probe-v1",
        "source_report": args.p107_report,
        "skip_forced_prefix_events": args.skip_forced_prefix_events,
        "production_baseline_tps": args.production_baseline_tps,
        "target_tps": args.target_tps,
        "draft_ms": draft_ms,
        "note": "Upper-bound policy from base-state P107 shadow traces, not recursive MTP rollout.",
        "decision": analysis["aggregate"]["recommendation"],
        "summary": {
            "routes_by_recommendation": routes_by_recommendation,
            "aggregate_top1_depth2": analysis["aggregate"]["metrics"]["top1"]["by_depth"]["depth2"]["projection"],
            "aggregate_top8_depth2": analysis["aggregate"]["metrics"]["top8"]["by_depth"]["depth2"]["projection"],
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
