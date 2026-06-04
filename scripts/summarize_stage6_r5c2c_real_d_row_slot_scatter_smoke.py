#!/usr/bin/env python3
"""Summarize Stage 6 R5-C2C real D-row slot scatter smoke artifacts."""
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
    schedules = data.get("scatter_schedules") or {}
    lines = [
        "# Stage 6 R5-C2C Real D-Row Slot Scatter Smoke Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Result | `{result_path}` |",
        f"| Decision | `{data.get('decision', 'unknown')}` |",
        f"| CUTLASS dir | `{data.get('cutlass_dir')}` |",
        f"| CUTLASS example | `{data.get('example')}` |",
        f"| Benchmark file | `{data.get('benchmark_file')}` |",
        f"| D-row digest file | `{data.get('d_row_digest_file')}` |",
        f"| Selected tokens/top_k/experts | `{shape.get('tokens')} / {shape.get('top_k')} / {shape.get('experts')}` |",
        f"| Tokens per expert | `{shape.get('tokens_per_expert')}` |",
        f"| Gate/up output width N | `{shape.get('n_gate_up')}` |",
        f"| Temporary D-row digest patch applied/restored | `{patch.get('applied')}` / `{patch.get('restored')}` |",
        f"| Real D-row slot scatter banked | `{passes.get('banked_real_d_row_slot_scatter')}` |",
        f"| Selected-output epilogue kernel banked | `{passes.get('banked_selected_output_kernel_epilogue')}` |",
        f"| SwiGLU/down projection banked | `{passes.get('banked_swiglu_or_down_projection')}` |",
        f"| Grouped-MoE FP4-MMA POC banked | `{passes.get('banked_grouped_moe_fp4_mma_poc')}` |",
        f"| Kernel speed banked | `{passes.get('banked_kernel_speed')}` |",
        f"| Default promotion banked | `{passes.get('banked_default_promotion')}` |",
        f"| Avg runtime ms (trace only) | `{parse.get('avg_runtime_ms')}` |",
        "",
        "## Run Gates",
        "",
        "| Gate | Value |",
        "|---|---:|",
    ]
    for key in [
        "build_invoked",
        "build_succeeded",
        "d_row_digest_patch_applied",
        "d_row_digest_patch_restored",
        "run_succeeded",
        "cooperative_passed",
        "pingpong_passed",
        "host_reference_seen",
        "dispositions_passed_count_ge_2",
        "groups_seen_match_experts",
        "tokens_per_expert_match",
        "grouped_order_complete",
        "digest_file_exists",
        "schedules_captured",
        "schedule_scatters_passed",
    ]:
        lines.append(f"| {key} | `{passes.get(key)}` |")

    lines.extend([
        "",
        "## Schedule Scatter Gates",
        "",
        "| Schedule | Records | Row counts | D/ref row digest match | Scatter match | Fault injections |",
        "|---|---:|---|---:|---:|---:|",
    ])
    for name, schedule in sorted(schedules.items()):
        sched_passes = schedule.get("passes") or {}
        lines.append(
            f"| `{name}` | `{schedule.get('records')}` | `{schedule.get('row_counts')}` | "
            f"`{sched_passes.get('d_ref_row_digests_match')}` | "
            f"`{sched_passes.get('scatter_d_ref_match')}` | "
            f"`{sched_passes.get('fault_injections_detected')}` |"
        )

    lines.extend([
        "",
        "## Boundary",
        "",
        "- This R5-C2C artifact banks only `banked_real_d_row_slot_scatter=true`.",
        "- It emits and scatters real CUTLASS D/ref row digests into `[T, top_k, N_gateup]` selected slots.",
        "- It does not bank an in-epilogue selected-output CUDA kernel.",
        "- It does not perform or bank SwiGLU activation, down projection, router validation, full grouped-MoE speed, server behavior, RC quality, or runtime default promotion.",
        "- Runtime/TFLOPS are trace-only; speed is first eligible at a later grouped active-MoE POC gate.",
        "",
    ])
    return "\n".join(lines)


def _strict_ok(data: dict[str, Any]) -> bool:
    passes = data.get("passes") or {}
    return bool(
        data.get("decision") == "PASS_R5C2C_REAL_D_ROW_SLOT_SCATTER_SMOKE"
        and passes.get("banked_real_d_row_slot_scatter") is True
        and passes.get("banked_selected_output_kernel_epilogue") is False
        and passes.get("banked_swiglu_or_down_projection") is False
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
