#!/usr/bin/env python3
"""P111: speculative decode throughput budget from P107 shadow reports.

P107 reports the MTP accept rate and draft-head cost, but the useful serving
question is stricter: does the accepted-token multiplier still win after paying
for the draft head? This script turns P107 trace summaries into a compact
throughput budget for both the measured research path and a target production
baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rate(summary: dict[str, Any], key: str) -> float | None:
    if key == "top1":
        return summary.get("accept_rate")
    ceiling = summary.get("topk_ceiling") or {}
    row = ceiling.get(key) or {}
    return row.get("rate")


def _budget_for_rate(*, rate: float, base_tps: float, draft_tps: float | None) -> dict[str, Any]:
    base_s = 1.0 / base_tps
    zero_overhead_tps = base_tps * (1.0 + rate)
    max_draft_s_for_155 = ((1.0 + rate) / 155.0) - base_s
    out: dict[str, Any] = {
        "accept_rate": rate,
        "zero_overhead_tps": zero_overhead_tps,
        "zero_overhead_speedup": 1.0 + rate,
        "feasible_even_zero_overhead_for_155": zero_overhead_tps >= 155.0,
        "max_draft_ms_for_155": (
            max_draft_s_for_155 * 1000.0 if max_draft_s_for_155 >= 0.0 else None
        ),
    }
    if draft_tps and draft_tps > 0:
        draft_s = 1.0 / draft_tps
        serial_tps = (1.0 + rate) / (base_s + draft_s)
        out.update(
            {
                "draft_tps": draft_tps,
                "draft_ms": draft_s * 1000.0,
                "serial_one_token_tps": serial_tps,
                "serial_one_token_speedup": serial_tps / base_tps,
                "draft_ms_fits_155_budget": (
                    max_draft_s_for_155 >= 0.0 and draft_s <= max_draft_s_for_155
                ),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p107-report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--production-baseline-tps", type=float, default=100.0)
    args = ap.parse_args()

    report = json.loads(Path(args.p107_report).read_text(encoding="utf-8"))
    summary = report["summary"]
    measured_base_tps = float(summary["decode_tps"])
    measured_draft_tps = float(summary["draft_tps"]) if summary.get("draft_tps") else None
    rates = {
        key: _rate(summary, key)
        for key in ("top1", "top2", "top4", "top8")
        if _rate(summary, key) is not None
    }
    measured = {
        key: _budget_for_rate(rate=float(rate), base_tps=measured_base_tps, draft_tps=measured_draft_tps)
        for key, rate in rates.items()
    }
    production = {
        key: _budget_for_rate(
            rate=float(rate),
            base_tps=args.production_baseline_tps,
            draft_tps=measured_draft_tps,
        )
        for key, rate in rates.items()
    }
    zero_overhead_top1_tps = production.get("top1", {}).get("zero_overhead_tps")
    result = {
        "schema_version": "lynn-p111-mtp-speculative-budget-v1",
        "source_report": args.p107_report,
        "decision": (
            "GREEN-BUDGET: top1 can clear 155 TPS with zero-overhead draft."
            if zero_overhead_top1_tps and zero_overhead_top1_tps >= 155.0
            else "AMBER-BUDGET: top1 cannot clear 155 TPS without either higher accept, hidden draft cost, or multi-token credit."
        ),
        "events": summary.get("events"),
        "measured_decode_tps": measured_base_tps,
        "measured_draft_tps": measured_draft_tps,
        "production_baseline_tps": args.production_baseline_tps,
        "rates": rates,
        "measured_path": measured,
        "production_projection": production,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
