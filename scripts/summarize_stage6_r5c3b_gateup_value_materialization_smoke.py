#!/usr/bin/env python3
"""Summarize Stage 6 R5-C3B gate/up value-materialization artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(data: dict[str, Any], result_path: Path) -> str:
    passes = data.get("passes") or {}
    shape = data.get("selected_expert_shape") or {}
    parse = data.get("run_parse") or {}
    patch = ((data.get("cutlass_run") or {}).get("patch") or {})
    schedules = data.get("value_schedules") or {}
    lines = [
        "# Stage 6 R5-C3B Gate/Up Value Materialization Smoke Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Result | `{result_path}` |",
        f"| Decision | `{data.get('decision', 'unknown')}` |",
        f"| Selected tokens/top_k/experts | `{shape.get('tokens')} / {shape.get('top_k')} / {shape.get('experts')}` |",
        f"| Tokens per expert | `{shape.get('tokens_per_expert')}` |",
        f"| Gate/up output width N | `{shape.get('n_gate_up')}` |",
        f"| Temporary D-row value patch applied/restored | `{patch.get('applied')}` / `{patch.get('restored')}` |",
        f"| Gate/up value materialization banked | `{passes.get('banked_gateup_value_materialization')}` |",
        f"| Host SwiGLU checksum smoke banked | `{passes.get('banked_host_swiglu_checksum_smoke')}` |",
        f"| Down projection numeric parity banked | `{passes.get('banked_down_projection_numeric_parity')}` |",
        f"| Grouped-MoE FP4-MMA POC banked | `{passes.get('banked_grouped_moe_fp4_mma_poc')}` |",
        f"| Kernel speed banked | `{passes.get('banked_kernel_speed')}` |",
        f"| Default promotion banked | `{passes.get('banked_default_promotion')}` |",
        f"| Avg runtime ms (trace only) | `{parse.get('avg_runtime_ms')}` |",
        "",
        "## Schedule Value Gates",
        "",
        "| Schedule | Records | Row counts | Value digest match | Scatter max abs | SwiGLU checksum | Fault injections |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for name, schedule in sorted(schedules.items()):
        sched_passes = schedule.get("passes") or {}
        lines.append(
            f"| `{name}` | `{schedule.get('records')}` | `{schedule.get('row_counts')}` | "
            f"`{sched_passes.get('value_digest_matches_r5c2c_digest')}` | "
            f"`{schedule.get('scatter_values_max_abs')}` | "
            f"`{schedule.get('host_swiglu_checksum')}` | "
            f"`{sched_passes.get('fault_injections_detected')}` |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "- This R5-C3B artifact banks only `banked_gateup_value_materialization=true` and `banked_host_swiglu_checksum_smoke=true`.",
        "- It emits full real CUTLASS D/ref row values and scatters them into `[T, top_k, N_gateup]` selected slots.",
        "- It does not bank down projection, weighted top-k reduction, full grouped-MoE speed, decode TPS, server/RC behavior, or runtime default promotion.",
        "",
    ])
    return "\n".join(lines)


def _strict_ok(data: dict[str, Any]) -> bool:
    passes = data.get("passes") or {}
    return bool(
        data.get("decision") == "PASS_R5C3B_GATEUP_VALUE_MATERIALIZATION_SMOKE"
        and passes.get("banked_gateup_value_materialization") is True
        and passes.get("banked_host_swiglu_checksum_smoke") is True
        and passes.get("banked_down_projection_numeric_parity") is False
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
