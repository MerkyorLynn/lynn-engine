#!/usr/bin/env python3
"""P109: offline reranker sanity check for MTP top-k shadow traces.

P107 can report that the verified next token is present in MTP top-k even when
the top-1 draft is wrong. This script checks whether a cheap margin-only
reranker could exploit that containment, before we spend runtime engineering on
multi-candidate decode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _events(report: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in report.get("rows", []):
        forced_len = 0
        if report.get("skip_forced_prefix_events"):
            forced_len = len((row.get("forced_prefix_report") or {}).get("ids") or [])
        for event in row.get("mtp_shadow", {}).get("trace", []):
            if int(event.get("step", 0)) < forced_len:
                continue
            event = dict(event)
            event["row_id"] = row.get("id")
            out.append(event)
    return out


def _top_ids(event: dict[str, Any]) -> list[int]:
    topk = event.get("draft_topk") or event.get("draft_top2") or {}
    return [int(x) for x in topk.get("ids", [])]


def _margin(event: dict[str, Any]) -> float:
    topk = event.get("draft_topk") or event.get("draft_top2") or {}
    return float(topk.get("top1_margin", 0.0))


def _coverage(events: list[dict[str, Any]], k: int) -> dict[str, Any]:
    usable = [event for event in events if len(_top_ids(event)) >= k]
    covered = sum(1 for event in usable if int(event["base_next_id"]) in _top_ids(event)[:k])
    return {
        "events": len(usable),
        "covered": covered,
        "rate": covered / len(usable) if usable else None,
    }


def _threshold_probe(events: list[dict[str, Any]], *, max_threshold: float, step: float) -> dict[str, Any]:
    best = {"threshold": 0.0, "accepted": -1, "rate": None, "switches": 0}
    n_steps = int(max_threshold / step)
    for idx in range(n_steps + 1):
        threshold = idx * step
        accepted = 0
        switches = 0
        for event in events:
            ids = _top_ids(event)
            if len(ids) < 2:
                continue
            use_second = _margin(event) <= threshold
            switches += int(use_second)
            picked = ids[1] if use_second else ids[0]
            accepted += int(picked == int(event["base_next_id"]))
        if accepted > int(best["accepted"]):
            best = {
                "threshold": threshold,
                "accepted": accepted,
                "rate": accepted / len(events) if events else None,
                "switches": switches,
            }
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p107-report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-threshold", type=float, default=8.0)
    ap.add_argument("--threshold-step", type=float, default=0.01)
    args = ap.parse_args()

    report = json.loads(Path(args.p107_report).read_text(encoding="utf-8"))
    events = _events(report)
    top1 = sum(1 for event in events if event.get("accepted"))
    top1_rate = top1 / len(events) if events else None
    threshold = _threshold_probe(events, max_threshold=args.max_threshold, step=args.threshold_step)
    result = {
        "schema_version": "lynn-p109-mtp-topk-reranker-probe-v1",
        "source_report": args.p107_report,
        "decision": (
            "AMBER: margin-only top2 reranker beats top1."
            if threshold["accepted"] > top1
            else "RED: margin-only top2 reranker does not beat top1."
        ),
        "events": len(events),
        "top1": {"accepted": top1, "rate": top1_rate},
        "coverage": {
            "top2": _coverage(events, 2),
            "top4": _coverage(events, 4),
            "top8": _coverage(events, 8),
        },
        "best_margin_top2_switch": threshold,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
