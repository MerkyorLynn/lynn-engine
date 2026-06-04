#!/usr/bin/env python3
"""Summarize Stage 6 P4C component profile artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "lynn-stage6-p4c-component-profile-v1"
PASS_DECISION = "PASS_P4C_COMPONENT_PROFILE_RECORDED"


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    bench = data.get("bench") or {}
    if data.get("schema") != SCHEMA:
        return "FAIL", "schema mismatch"
    if data.get("decision") != PASS_DECISION:
        return "FAIL", "top-level decision mismatch"
    if data.get("banked_p4c_component_profile") is not True:
        return "FAIL", "P4C component profile was not banked"
    if data.get("banked_fused_kernel") is not False:
        return "FAIL", "fused-kernel speed boundary violated"
    if data.get("banked_default_promotion") is not False:
        return "FAIL", "default promotion boundary violated"
    for gate in ("numeric_vs_reference", "timing_recorded", "promotion_boundary_closed", "all"):
        if passes.get(gate) is not True:
            return "FAIL", f"{gate} gate fail"
    if not (bench.get("full_p4c_active_reuse_contract") or {}).get("median_us"):
        return "FAIL", "missing full P4C timing"
    if not (bench.get("component_gate_up_allocating") or {}).get("median_us"):
        return "FAIL", "missing gate/up timing"
    if not (bench.get("component_down_allocating") or {}).get("median_us"):
        return "FAIL", "missing down timing"
    return "PASS", "P4C component profile recorded; promotion still closed"


def _diff_line(numeric: dict[str, Any], name: str) -> tuple[Any, Any]:
    diff = numeric.get(name) or {}
    return diff.get("rel_l2"), diff.get("max_abs")


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    bench = data.get("bench") or {}
    full = bench.get("full_p4c_active_reuse_contract") or {}
    gate = bench.get("component_gate_up_allocating") or {}
    down = bench.get("component_down_allocating") or {}
    numeric = data.get("numeric_vs_reference") or {}
    gate_rel, gate_abs = _diff_line(numeric, "gate_inter_scratch")
    down_rel, down_abs = _diff_line(numeric, "down_on_ref_scratch")
    comp_rel, comp_abs = _diff_line(numeric, "gate_plus_down_composed")
    lines = [
        "# Stage 6 P4C Component Profile Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Decision | `{data.get('decision')}` |",
        f"| Device | `{data.get('device_name', 'unknown')}` |",
        f"| Capability | `{data.get('capability', 'unknown')}` |",
        f"| Banked component profile | `{data.get('banked_p4c_component_profile')}` |",
        f"| Banked fused kernel speed | `{data.get('banked_fused_kernel')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        f"| Full P4C median | `{full.get('median_us')}` us |",
        f"| Gate/up component median | `{gate.get('median_us')}` us |",
        f"| Down component median | `{down.get('median_us')}` us |",
        f"| Component sum | `{bench.get('component_sum_us')}` us |",
        f"| Gate share | `{bench.get('gate_share_of_component_sum')}` |",
        f"| Down share | `{bench.get('down_share_of_component_sum')}` |",
        f"| Component sum / full | `{bench.get('component_sum_vs_full_ratio')}` |",
        f"| Gate rel L2 / max abs | `{gate_rel}` / `{gate_abs}` |",
        f"| Down rel L2 / max abs | `{down_rel}` / `{down_abs}` |",
        f"| Composed rel L2 / max abs | `{comp_rel}` / `{comp_abs}` |",
        f"| Caveat | `{data.get('component_timing_caveat')}` |",
        "",
        "## Boundary",
        "",
        "- This banks only `banked_p4c_component_profile=true`.",
        "- Component timings use existing allocation-returning symbols and are diagnostic only.",
        "- The next speed candidate should target the larger component first, then rerun the P4C speed baseline and RC gates.",
    ]
    error = data.get("load_error_tail")
    if error:
        lines.extend(["", "## Error Tail", "", "```text", str(error)[-1200:], "```"])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json")
    ap.add_argument("--markdown-out", default="")
    ap.add_argument("--strict-exit", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.result_json).read_text())
    md = summarize(data)
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
    sys.stdout.write(md)
    verdict, _ = _verdict(data)
    return 0 if (verdict == "PASS" or not args.strict_exit) else 2


if __name__ == "__main__":
    raise SystemExit(main())
