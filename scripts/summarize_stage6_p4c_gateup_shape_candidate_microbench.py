#!/usr/bin/env python3
"""Summarize Stage 6 P4C gate/up shape-candidate microbench artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "lynn-stage6-p4c-gateup-shape-candidate-microbench-v1"
PASS_DECISION = "PASS_P4C_GATEUP_SHAPE_CANDIDATE_RECORDED"


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    bench = data.get("bench") or {}
    if data.get("schema") != SCHEMA:
        return "FAIL", "schema mismatch"
    if data.get("decision") != PASS_DECISION:
        return "FAIL", "top-level decision mismatch"
    if data.get("banked_p4c_gateup_shape_candidate") is not True:
        return "FAIL", "P4C gate/up shape candidate was not banked"
    if data.get("banked_fused_kernel") is not False:
        return "FAIL", "fused-kernel speed boundary violated"
    if data.get("banked_default_promotion") is not False:
        return "FAIL", "default promotion boundary violated"
    for gate in (
        "numeric_vs_reference",
        "timing_recorded",
        "candidate_speed_floor",
        "zero_bf16_shadow_weight_abi",
        "active_scratch_reuse_abi",
        "packed_byte_budget",
        "promotion_boundary_closed",
        "all",
    ):
        if passes.get(gate) is not True:
            return "FAIL", f"{gate} gate fail"
    for name in (
        "current_p4c_active_reuse_contract",
        "candidate_p4c_active_reuse_contract",
        "reference_p4a_current_tile",
        "reference_p4a_candidate_tile",
    ):
        if not (bench.get(name) or {}).get("median_us"):
            return "FAIL", f"missing {name} timing"
    return "PASS", "P4C gate/up shape candidate recorded; default promotion still closed"


def _diff_line(numeric: dict[str, Any], name: str) -> tuple[Any, Any]:
    diff = numeric.get(name) or {}
    return diff.get("rel_l2"), diff.get("max_abs")


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    bench = data.get("bench") or {}
    numeric = data.get("numeric_vs_reference") or {}
    cand_out_rel, cand_out_abs = _diff_line(numeric, "candidate_vs_p4a_candidate_tile_out")
    cand_inter_rel, cand_inter_abs = _diff_line(numeric, "candidate_vs_p4a_candidate_tile_inter_scratch")
    current = bench.get("current_p4c_active_reuse_contract") or {}
    candidate = bench.get("candidate_p4c_active_reuse_contract") or {}
    ref_current = bench.get("reference_p4a_current_tile") or {}
    ref_candidate = bench.get("reference_p4a_candidate_tile") or {}
    lines = [
        "# Stage 6 P4C Gate/Up Shape Candidate Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Decision | `{data.get('decision')}` |",
        f"| Device | `{data.get('device_name', 'unknown')}` |",
        f"| Capability | `{data.get('capability', 'unknown')}` |",
        f"| Current tile_inter | `{data.get('current_tile_inter')}` |",
        f"| Candidate tile_inter | `{data.get('candidate_tile_inter')}` |",
        f"| Current P4C median | `{current.get('median_us')}` us |",
        f"| Candidate P4C median | `{candidate.get('median_us')}` us |",
        f"| Candidate speedup vs current | `{bench.get('candidate_vs_current_speedup')}` |",
        f"| Candidate - current | `{bench.get('candidate_minus_current_us')}` us |",
        f"| P4A current tile median | `{ref_current.get('median_us')}` us |",
        f"| P4A candidate tile median | `{ref_candidate.get('median_us')}` us |",
        f"| P4A candidate/current speedup | `{bench.get('reference_candidate_vs_current_speedup')}` |",
        f"| Candidate out rel L2 / max abs | `{cand_out_rel}` / `{cand_out_abs}` |",
        f"| Candidate scratch rel L2 / max abs | `{cand_inter_rel}` / `{cand_inter_abs}` |",
        f"| Banked P4C gate/up shape candidate | `{data.get('banked_p4c_gateup_shape_candidate')}` |",
        f"| Banked fused kernel speed | `{data.get('banked_fused_kernel')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        "",
        "## Boundary",
        "",
        "- This banks only `banked_p4c_gateup_shape_candidate=true`.",
        "- It does not bank fused-kernel speed or default promotion.",
        "- If this remains faster in server/RC context, wire it as an opt-in runtime default candidate and rerun quality gates.",
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
