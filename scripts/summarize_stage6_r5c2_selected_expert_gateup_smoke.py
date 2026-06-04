#!/usr/bin/env python3
"""Summarize Stage 6 R5-C2 selected-expert gate/up smoke artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(data: dict[str, Any], result_path: Path) -> str:
    passes = data.get("passes") or {}
    git = data.get("git") or {}
    head = ((git.get("head") or {}).get("stdout_tail") or (git.get("head") or {}).get("stdout") or "unknown").strip()
    branch = ((git.get("branch") or {}).get("stdout_tail") or (git.get("branch") or {}).get("stdout") or "unknown").strip()
    shape = data.get("selected_expert_shape") or {}
    parse = data.get("run_parse") or {}
    build = data.get("build_result") or {}
    patch = (build.get("atomic_scope_patch") or {}) if isinstance(build, dict) else {}
    lines = [
        "# Stage 6 R5-C2 Selected-Expert Gate/Up Smoke Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Result | `{result_path}` |",
        f"| Decision | `{data.get('decision', 'unknown')}` |",
        f"| CUTLASS dir | `{data.get('cutlass_dir')}` |",
        f"| CUTLASS git | `{head}` (`{branch}`) |",
        f"| Example | `{data.get('example')}` |",
        f"| Benchmark file | `{data.get('benchmark_file')}` |",
        f"| Selected tokens/top_k/experts | `{shape.get('tokens')} / {shape.get('top_k')} / {shape.get('experts')}` |",
        f"| Tokens per expert | `{shape.get('tokens_per_expert')}` |",
        f"| Gate/up shape | `M=tokens_per_expert[e] N={shape.get('n_gate_up')} K={shape.get('k_hidden')}` |",
        f"| Groups seen | `{data.get('groups_seen')}` |",
        f"| Temporary CUDA atomic patch applied/restored | `{patch.get('applied')}` / `{patch.get('restored')}` |",
        f"| Selected-expert gate/up smoke banked | `{passes.get('banked_selected_expert_gate_up_smoke')}` |",
        f"| Grouped-MoE FP4-MMA POC banked | `{passes.get('banked_grouped_moe_fp4_mma_poc')}` |",
        f"| Kernel speed banked | `{passes.get('banked_kernel_speed')}` |",
        f"| Default promotion banked | `{passes.get('banked_default_promotion')}` |",
        "",
        "## Run Gates",
        "",
        "| Gate | Value |",
        "|---|---:|",
        f"| Route tokens match | `{passes.get('route_tokens_match')}` |",
        f"| Route top-k unique | `{passes.get('route_topk_unique')}` |",
        f"| Tokens per expert match | `{passes.get('tokens_per_expert_match')}` |",
        f"| Benchmark shapes aligned to 32 | `{passes.get('benchmark_shapes_aligned_32')}` |",
        f"| Benchmark groups match experts | `{passes.get('benchmark_groups_match_experts')}` |",
        f"| Groups seen match experts | `{passes.get('groups_seen_match_experts')}` |",
        f"| Cooperative schedule passed | `{passes.get('cooperative_passed')}` |",
        f"| Pingpong schedule passed | `{passes.get('pingpong_passed')}` |",
        f"| Host reference seen | `{passes.get('host_reference_seen')}` |",
        f"| Disposition passed count >= 2 | `{passes.get('dispositions_passed_count_ge_2')}` |",
        f"| Avg runtime ms | `{parse.get('avg_runtime_ms')}` |",
        f"| TFLOPS | `{parse.get('tflops')}` |",
        "",
        "## Boundary",
        "",
        "- This R5-C2 artifact banks only `banked_selected_expert_gate_up_smoke=true`.",
        "- It maps `tokens_per_expert` to CUTLASS 79d per-group `M` shapes and checks host-reference numeric correctness.",
        "- It does not bank Lynn slot-preserving gather/scatter, down projection, full grouped-MoE speed, kernel speed, or runtime default promotion.",
        "- The next gate is R5-C2B slot-preserving selected-output bridge, not R5-C3 full grouped-MoE speed.",
        "",
    ]
    return "\n".join(lines)


def _strict_ok(data: dict[str, Any]) -> bool:
    passes = data.get("passes") or {}
    return bool(
        data.get("decision") == "PASS_R5C2_SELECTED_EXPERT_GATEUP_NUMERIC_SMOKE"
        and passes.get("banked_selected_expert_gate_up_smoke") is True
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
