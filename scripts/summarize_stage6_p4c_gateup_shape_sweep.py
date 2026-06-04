#!/usr/bin/env python3
"""Summarize Stage 6 P4C gate/up launch-shape sweep artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "lynn-stage6-p4c-gateup-shape-sweep-v1"
PASS_DECISION = "PASS_P4C_GATEUP_SHAPE_SWEEP_RECORDED"


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    bench = data.get("bench") or {}
    if data.get("schema") != SCHEMA:
        return "FAIL", "schema mismatch"
    if data.get("decision") != PASS_DECISION:
        return "FAIL", "top-level decision mismatch"
    if data.get("banked_p4c_gateup_shape_sweep") is not True:
        return "FAIL", "P4C gate/up shape sweep was not banked"
    if data.get("banked_p4c_gateup_candidate") is not False:
        return "FAIL", "gate/up candidate boundary violated"
    if data.get("banked_fused_kernel") is not False:
        return "FAIL", "fused-kernel speed boundary violated"
    if data.get("banked_default_promotion") is not False:
        return "FAIL", "default promotion boundary violated"
    for gate in (
        "baseline_numeric_vs_reference",
        "variants_numeric_vs_reference",
        "timing_recorded",
        "promotion_boundary_closed",
        "all",
    ):
        if passes.get(gate) is not True:
            return "FAIL", f"{gate} gate fail"
    if not (bench.get("current_baseline_gate_up") or {}).get("median_us"):
        return "FAIL", "missing current gate/up baseline timing"
    if not bench.get("variants"):
        return "FAIL", "missing shape variants"
    if not bench.get("best_variant"):
        return "FAIL", "missing best variant"
    return "PASS", "P4C gate/up launch-shape sweep recorded; promotion still closed"


def _diff_line(diff: dict[str, Any] | None) -> tuple[Any, Any]:
    diff = diff or {}
    return diff.get("rel_l2"), diff.get("max_abs")


def _variant_rows(variants: list[dict[str, Any]]) -> list[str]:
    rows = ["| Shape | Median us | Speedup vs current | Numeric ok | rel L2 / max abs |", "|---|---:|---:|---|---|"]
    for variant in sorted(variants, key=lambda item: (item.get("tile_inter", 0), item.get("threads", 0))):
        rel_l2, max_abs = _diff_line(variant.get("diff"))
        rows.append(
            f"| `{variant.get('key')}` | `{variant.get('median_us')}` | "
            f"`{variant.get('speedup_vs_current')}` | `{variant.get('numeric_ok')}` | "
            f"`{rel_l2}` / `{max_abs}` |"
        )
    return rows


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    bench = data.get("bench") or {}
    baseline = bench.get("current_baseline_gate_up") or {}
    best = bench.get("best_variant") or {}
    numeric = data.get("numeric_vs_reference") or {}
    base_rel, base_abs = _diff_line(numeric.get("current_baseline_gate_up"))
    variants = bench.get("variants") or []
    lines = [
        "# Stage 6 P4C Gate/Up Shape Sweep Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Decision | `{data.get('decision')}` |",
        f"| Device | `{data.get('device_name', 'unknown')}` |",
        f"| Capability | `{data.get('capability', 'unknown')}` |",
        f"| Banked shape sweep | `{data.get('banked_p4c_gateup_shape_sweep')}` |",
        f"| Banked gate/up candidate | `{data.get('banked_p4c_gateup_candidate')}` |",
        f"| Banked fused kernel speed | `{data.get('banked_fused_kernel')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        f"| Current baseline symbol | `{data.get('baseline_symbol')}` |",
        f"| Variant symbol | `{data.get('variant_symbol')}` |",
        f"| Baseline tile_inter | `{data.get('baseline_tile_inter')}` |",
        f"| Baseline median | `{baseline.get('median_us')}` us |",
        f"| Baseline rel L2 / max abs | `{base_rel}` / `{base_abs}` |",
        f"| Best shape | `{best.get('key')}` |",
        f"| Best median | `{best.get('median_us')}` us |",
        f"| Best speedup vs current | `{bench.get('best_speedup_vs_current')}` |",
        f"| Best actionable >= floor | `{bench.get('best_is_actionable')}` "
        f"(floor `{data.get('actionable_speedup_floor')}`) |",
        f"| Caveat | `{data.get('component_timing_caveat')}` |",
        "",
        "## Shape Sweep",
        "",
        *_variant_rows(variants),
        "",
        "## Boundary",
        "",
        "- This banks only `banked_p4c_gateup_shape_sweep=true`.",
        "- It does not bank a gate/up speed candidate, fused kernel speed, or default promotion.",
        "- If `best_is_actionable=false`, scalar launch-shape tuning is exhausted and the next cut should be a real CUDA/CUTLASS gate/up kernel.",
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
