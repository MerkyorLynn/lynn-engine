#!/usr/bin/env python3
"""Summarize Stage 6 R5-C2 MoE-shape census artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(data: dict[str, Any], result_path: Path) -> str:
    passes = data.get("passes") or {}
    git = data.get("git") or {}
    head = ((git.get("head") or {}).get("stdout") or "unknown").strip()
    branch = ((git.get("branch") or {}).get("stdout") or "unknown").strip()
    lines = [
        "# Stage 6 R5-C2 MoE Shape Census Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Result | `{result_path}` |",
        f"| Decision | `{data.get('decision', 'unknown')}` |",
        f"| CUTLASS dir | `{data.get('cutlass_dir')}` |",
        f"| CUTLASS git | `{head}` (`{branch}`) |",
        f"| MoE shape census banked | `{passes.get('banked_moe_shape_census')}` |",
        f"| Requires new minimal harness | `{passes.get('requires_new_minimal_harness')}` |",
        f"| Selected expert gate/up smoke banked | `{passes.get('banked_selected_expert_gate_up_smoke')}` |",
        f"| Grouped-MoE FP4-MMA POC banked | `{passes.get('banked_grouped_moe_fp4_mma_poc')}` |",
        f"| Kernel speed banked | `{passes.get('banked_kernel_speed')}` |",
        f"| Default promotion banked | `{passes.get('banked_default_promotion')}` |",
        "",
        "## Key Source Split",
        "",
        "| Source | Evidence |",
        "|---|---|",
        "| CUTLASS 79d | SM120 native NVF4+UE4M3 generic grouped GEMM; lacks `MoEProblemShape` and `tokens_per_expert`. |",
        "| CUTLASS 92 | Has `MoEProblemShape` + `tokens_per_expert` and NVF4+UE4M3, but uses Sm100 schedules. |",
        "",
        "## Boundary",
        "",
        "- This artifact banks only `banked_moe_shape_census=true`.",
        "- It does not bank selected-expert gate/up numeric smoke, grouped-MoE speed, or default promotion.",
        "- R5-C2 implementation must combine 92-style MoE shape semantics with 79d-style SM120 execution.",
        "",
    ]
    return "\n".join(lines)


def _strict_ok(data: dict[str, Any]) -> bool:
    passes = data.get("passes") or {}
    return bool(
        data.get("decision") == "PASS_R5C2_MOE_SHAPE_CENSUS_NEW_HARNESS_REQUIRED"
        and passes.get("banked_moe_shape_census") is True
        and passes.get("requires_new_minimal_harness") is True
        and passes.get("banked_selected_expert_gate_up_smoke") is False
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
