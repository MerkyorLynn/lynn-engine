#!/usr/bin/env python3
"""Summarize Stage 6 R5-C4 full active-MoE prefill speed A/B artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PASS_DECISION = "PASS_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB"
DIAGNOSTIC_DECISION = "DIAGNOSTIC_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_CLOSED"


def _passes(data: dict[str, Any]) -> dict[str, Any]:
    passes = data.get("passes")
    return passes if isinstance(passes, dict) else {}


def _lanes(data: dict[str, Any]) -> dict[str, Any]:
    lanes = data.get("lanes")
    return lanes if isinstance(lanes, dict) else {}


def _strict_pass(data: dict[str, Any]) -> bool:
    passes = _passes(data)
    return bool(
        data.get("decision") == PASS_DECISION
        and passes.get("input_r5c3c_passed") is True
        and passes.get("same_scope_ab") is True
        and passes.get("real_model_weights") is True
        and passes.get("real_router_outputs") is True
        and passes.get("candidate_no_active_bf16_shadow") is True
        and passes.get("candidate_no_reload") is True
        and passes.get("candidate_no_bf16_weight_materialization") is True
        and passes.get("candidate_full_active_moe_boundary_timed") is True
        and passes.get("timing_includes_gateup_swiglu_down_weighted_scatter") is True
        and passes.get("numeric_vs_w4a16_or_p3_reference") is True
        and passes.get("candidate_median_speedup_vs_best_reference_ge_1p05") is True
        and passes.get("banked_full_active_moe_prefill_speed") is True
        and passes.get("banked_grouped_moe_fp4_mma_poc") is True
        and passes.get("banked_kernel_speed") is True
        and passes.get("banked_decode_tps") is False
        and passes.get("banked_server_rc") is False
        and passes.get("banked_default_promotion") is False
        and passes.get("banked_full_transformer_prefill") is False
    )


def _strict_diagnostic(data: dict[str, Any]) -> bool:
    passes = _passes(data)
    return bool(
        data.get("decision") == DIAGNOSTIC_DECISION
        and passes.get("input_r5c3c_passed") is True
        and passes.get("same_scope_ab") is True
        and passes.get("numeric_vs_w4a16_or_p3_reference") is True
        and passes.get("banked_full_active_moe_prefill_speed") is False
        and passes.get("banked_grouped_moe_fp4_mma_poc") is False
        and passes.get("banked_kernel_speed") is False
        and passes.get("banked_decode_tps") is False
        and passes.get("banked_server_rc") is False
        and passes.get("banked_default_promotion") is False
        and passes.get("banked_full_transformer_prefill") is False
    )


def summarize(data: dict[str, Any], result_path: Path) -> str:
    passes = _passes(data)
    lanes = _lanes(data)
    lines = [
        "# Stage 6 R5-C4 Full Active-MoE Prefill Speed A/B Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Result | `{result_path}` |",
        f"| Decision | `{data.get('decision', 'unknown')}` |",
        f"| Kernel speed scope | `{data.get('kernel_speed_scope', 'unknown')}` |",
        f"| Input R5-C3C passed | `{passes.get('input_r5c3c_passed')}` |",
        f"| Same-scope A/B | `{passes.get('same_scope_ab')}` |",
        f"| Real model weights/router outputs | `{passes.get('real_model_weights')}` / `{passes.get('real_router_outputs')}` |",
        f"| No active BF16 shadow/reload/materialization | `{passes.get('candidate_no_active_bf16_shadow')}` / `{passes.get('candidate_no_reload')}` / `{passes.get('candidate_no_bf16_weight_materialization')}` |",
        f"| Full active-MoE boundary timed | `{passes.get('candidate_full_active_moe_boundary_timed')}` |",
        f"| Timing includes gateup/SwiGLU/down/weighted/scatter | `{passes.get('timing_includes_gateup_swiglu_down_weighted_scatter')}` |",
        f"| Numeric parity banked | `{passes.get('numeric_vs_w4a16_or_p3_reference')}` |",
        f"| Best-reference speed gate | `{passes.get('candidate_median_speedup_vs_best_reference_ge_1p05')}` |",
        f"| Full active-MoE speed banked | `{passes.get('banked_full_active_moe_prefill_speed')}` |",
        f"| Grouped-MoE FP4-MMA POC banked | `{passes.get('banked_grouped_moe_fp4_mma_poc')}` |",
        f"| Kernel speed banked | `{passes.get('banked_kernel_speed')}` |",
        f"| Decode TPS banked | `{passes.get('banked_decode_tps')}` |",
        f"| Server/RC banked | `{passes.get('banked_server_rc')}` |",
        f"| Default promotion banked | `{passes.get('banked_default_promotion')}` |",
        f"| Full transformer prefill banked | `{passes.get('banked_full_transformer_prefill')}` |",
        "",
        "## Lanes",
        "",
        "| Lane | Candidate ms | Best reference ms | Speedup vs best | Numeric max_abs | Numeric rel_l2 | Numeric cosine |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, lane in sorted(lanes.items()):
        lines.append(
            f"| `{name}` | `{lane.get('candidate_ms')}` | `{lane.get('best_reference_ms')}` | "
            f"`{lane.get('median_speedup_vs_best_reference')}` | `{lane.get('numeric_max_abs')}` | "
            f"`{lane.get('numeric_rel_l2')}` | `{lane.get('numeric_cosine')}` |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- A PASS banks only full active-MoE prefill speed for the declared R6000 lane.",
            "- It does not bank Spark decode TPS, server/RC behavior, default promotion, or full transformer prefill.",
            "- Gate/up-only timing traces and host composition parity artifacts must fail strict PASS.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result_json")
    ap.add_argument("--markdown-out", default="")
    ap.add_argument("--strict-pass-exit", action="store_true")
    ap.add_argument("--strict-diagnostic-exit", action="store_true")
    args = ap.parse_args()
    result_path = Path(args.result_json)
    data = json.loads(result_path.read_text(encoding="utf-8"))
    md = summarize(data, result_path)
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n", encoding="utf-8")
    print(md)
    if args.strict_pass_exit and not _strict_pass(data):
        return 2
    if args.strict_diagnostic_exit and not _strict_diagnostic(data):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
