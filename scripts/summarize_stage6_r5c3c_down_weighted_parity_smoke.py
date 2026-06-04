#!/usr/bin/env python3
"""Summarize Stage 6 R5-C3C down + weighted top-k parity artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(data: dict[str, Any], result_path: Path) -> str:
    passes = data.get("passes") or {}
    shape = data.get("selected_expert_shape") or {}
    schedules = data.get("schedule_parity") or {}
    lines = [
        "# Stage 6 R5-C3C Down + Weighted Top-K Parity Smoke Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Result | `{result_path}` |",
        f"| Decision | `{data.get('decision', 'unknown')}` |",
        f"| Input R5-C3B decision | `{data.get('input_decision')}` |",
        f"| Selected tokens/top_k/experts | `{shape.get('tokens')} / {shape.get('top_k')} / {shape.get('experts')}` |",
        f"| SwiGLU hidden / down out dim | `{shape.get('swiglu_hidden')} / {shape.get('down_out_dim')}` |",
        f"| Down projection numeric parity banked | `{passes.get('banked_down_projection_numeric_parity')}` |",
        f"| Weighted top-k numeric parity banked | `{passes.get('banked_weighted_topk_numeric_parity')}` |",
        f"| Grouped-MoE FP4-MMA POC banked | `{passes.get('banked_grouped_moe_fp4_mma_poc')}` |",
        f"| Kernel speed banked | `{passes.get('banked_kernel_speed')}` |",
        f"| Default promotion banked | `{passes.get('banked_default_promotion')}` |",
        "",
        "## Schedule Parity Gates",
        "",
        "| Schedule | Records | SwiGLU max abs | Down max abs | Weighted max abs | Weighted hash match | Fault injections |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, schedule in sorted(schedules.items()):
        sched_passes = schedule.get("passes") or {}
        lines.append(
            f"| `{name}` | `{schedule.get('records')}` | "
            f"`{schedule.get('swiglu_d_ref_max_abs')}` | "
            f"`{schedule.get('down_d_ref_max_abs')}` | "
            f"`{schedule.get('weighted_topk_d_ref_max_abs')}` | "
            f"`{sched_passes.get('weighted_topk_hash_match')}` | "
            f"`{sched_passes.get('fault_injections_detected')}` |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "- This R5-C3C artifact banks only `banked_down_projection_numeric_parity=true` and `banked_weighted_topk_numeric_parity=true`.",
        "- It consumes real R5-C3B CUTLASS gate/up D/ref values, then runs host SwiGLU, deterministic down projection, and route-weighted top-k reduction.",
        "- It does not bank full active-MoE FP4-MMA speed, decode TPS, server/RC behavior, or runtime default promotion.",
        "",
    ])
    return "\n".join(lines)


def _strict_ok(data: dict[str, Any]) -> bool:
    passes = data.get("passes") or {}
    return bool(
        data.get("decision") == "PASS_R5C3C_DOWN_WEIGHTED_PARITY_SMOKE"
        and passes.get("banked_down_projection_numeric_parity") is True
        and passes.get("banked_weighted_topk_numeric_parity") is True
        and passes.get("banked_grouped_moe_fp4_mma_poc") is False
        and passes.get("banked_kernel_speed") is False
        and passes.get("banked_default_promotion") is False
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result_json")
    ap.add_argument("--markdown-out", default="")
    ap.add_argument("--strict-exit", action="store_true")
    args = ap.parse_args()
    result_path = Path(args.result_json)
    data = json.loads(result_path.read_text(encoding="utf-8"))
    md = summarize(data, result_path)
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n", encoding="utf-8")
    print(md)
    return 2 if args.strict_exit and not _strict_ok(data) else 0


if __name__ == "__main__":
    raise SystemExit(main())
