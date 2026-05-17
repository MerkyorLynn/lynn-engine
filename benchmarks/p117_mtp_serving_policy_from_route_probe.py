#!/usr/bin/env python3
"""P117: compact serving-policy artefact from P116 route-policy probes.

P116 is intentionally verbose because it preserves per-route depth/top-k
budgets. P117 distills that into a policy file a runtime implementation can use
as its first allowlist/denylist scaffold.

The output is still research policy, not an enabled speculative decoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _recommendation_key(row: dict[str, Any]) -> str:
    return str(row["recommendation"]).split(":", 1)[0]


def _route_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list((report.get("analysis") or {}).get("routes") or [])


def _route_policy(report: dict[str, Any], *, name: str) -> dict[str, Any]:
    groups: dict[str, list[str]] = {}
    route_metrics: dict[str, Any] = {}
    for row in _route_rows(report):
        route_id = str(row["id"])
        key = _recommendation_key(row)
        groups.setdefault(key, []).append(route_id)
        metrics = row.get("metrics") or {}
        route_metrics[route_id] = {
            "recommendation": key,
            "events": row.get("events"),
            "top1_depth2_same_cost_tps": (
                metrics.get("top1", {})
                .get("by_depth", {})
                .get("depth2", {})
                .get("projection", {})
                .get("optimistic_same_draft_cost_tps")
            ),
            "top8_depth2_same_cost_tps": (
                metrics.get("top8", {})
                .get("by_depth", {})
                .get("depth2", {})
                .get("projection", {})
                .get("optimistic_same_draft_cost_tps")
            ),
        }

    summary = report.get("summary") or {}
    return {
        "name": name,
        "source_report": report.get("source_report"),
        "decision": report.get("decision"),
        "draft_ms": report.get("draft_ms"),
        "target_tps": report.get("target_tps"),
        "groups": {key: sorted(value) for key, value in sorted(groups.items())},
        "route_metrics": route_metrics,
        "aggregate_top1_depth2": summary.get("aggregate_top1_depth2"),
        "aggregate_top8_depth2": summary.get("aggregate_top8_depth2"),
    }


def _runtime_plan(raw_policy: dict[str, Any], structured_policy: dict[str, Any] | None) -> dict[str, Any]:
    raw_groups = raw_policy.get("groups") or {}
    structured_groups = (structured_policy or {}).get("groups") or {}
    top1_routes = sorted(raw_groups.get("ENABLE_TOP1_DEPTH2", []))
    overlap_routes = sorted(raw_groups.get("TOP1_DEPTH2_NEEDS_OVERLAP", []))
    top8_routes = sorted(raw_groups.get("TOP8_DEPTH2_RERANK_CANDIDATE", []))
    disabled_routes = sorted(
        set(raw_groups.get("DISABLE_OR_RETRAIN", []))
        | set(raw_groups.get("TOP8_DEPTH4_ONLY", []))
    )
    structured_disabled = sorted(
        route
        for key, routes in structured_groups.items()
        if key in {"DISABLE_OR_RETRAIN", "TOP8_DEPTH4_ONLY"}
        for route in routes
    )
    return {
        "default": "disabled_until_route_classified",
        "env": {
            "LYNN_MTP_LAYER_MOE": "decode_slot_sorted",
            "LYNN_MTP_SPEC_POLICY": "route_allowlist_v1",
            "LYNN_MTP_DISABLE_FOR_FORMAT_GUARD": "1",
        },
        "raw_routes": {
            "enable_top1_depth2_first": top1_routes,
            "requires_overlap_before_enable": overlap_routes,
            "top8_rerank_research": top8_routes,
            "disabled_or_retrain": disabled_routes,
        },
        "structured_guarded_routes": {
            "default": "disabled",
            "disabled_or_retrain": structured_disabled,
            "research_only": sorted(
                route
                for key, routes in structured_groups.items()
                if key not in {"DISABLE_OR_RETRAIN", "TOP8_DEPTH4_ONLY"}
                for route in routes
            ),
        },
        "promotion_gates": {
            "top1_depth2_route": "same-cost projected TPS >= 155 and route exact/format gate GREEN",
            "top8_depth2_route": "reranker/verifier must improve accept without changing greedy text",
            "structured_guarded": "top8 depth2 zero-overhead > 155 and route policy no longer dominated by DISABLE_OR_RETRAIN",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-p116", required=True)
    ap.add_argument("--structured-p116")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw_report = _load(args.raw_p116)
    structured_report = _load(args.structured_p116) if args.structured_p116 else None
    raw_policy = _route_policy(raw_report, name="raw_v34_slotsorted")
    structured_policy = (
        _route_policy(structured_report, name="structured_v4_forced_skip_slotsorted")
        if structured_report
        else None
    )
    result = {
        "schema_version": "lynn-p117-mtp-serving-policy-v1",
        "decision": (
            "AMBER-POLICY: route allowlist is ready for a prototype, but global MTP remains disabled."
        ),
        "note": "Policy is derived from upper-bound P107/P116 traces; it does not enable speculative decode by itself.",
        "raw": raw_policy,
        "structured_guarded": structured_policy,
        "runtime_plan": _runtime_plan(raw_policy, structured_policy),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
