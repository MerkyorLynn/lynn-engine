#!/usr/bin/env python3
"""Summarize Stage 6 R5-C CUTLASS UE4M3 ABI census artifacts."""
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
        "# Stage 6 R5-C CUTLASS UE4M3 Census Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Result | `{result_path}` |",
        f"| Decision | `{data.get('decision', 'unknown')}` |",
        f"| CUTLASS dir | `{data.get('cutlass_dir')}` |",
        f"| CUTLASS git | `{head}` (`{branch}`) |",
        f"| CUTLASS ABI banked | `{passes.get('banked_cutlass_abi')}` |",
        f"| Grouped-MoE FP4-MMA POC banked | `{passes.get('banked_grouped_moe_fp4_mma_poc')}` |",
        f"| Kernel speed banked | `{passes.get('banked_kernel_speed')}` |",
        f"| Default promotion banked | `{passes.get('banked_default_promotion')}` |",
        "",
        "## Required Tokens",
        "",
        "| Gate | Pass |",
        "|---|---:|",
    ]
    for key in [
        "sm120_ue4m3_macro_seen",
        "scale_format_ue4m3_seen",
        "scale_type_ue4m3_seen",
        "mxf4_e2m1_format_seen",
        "sm120_e2m1_ue4m3_specialization_seen",
        "sm120_mxf4nvf4_ue4m3_asm_seen",
        "expected_examples_seen",
        "sm120_tests_seen",
    ]:
        lines.append(f"| `{key}` | `{passes.get(key)}` |")
    lines.extend([
        "",
        "## Evidence Snippets",
        "",
        "| Source | Line | Text |",
        "|---|---:|---|",
    ])
    for group, rows in (data.get("token_hits") or {}).items():
        for row in rows[:8]:
            text = str(row.get("text", "")).replace("|", "\\|")
            lines.append(f"| `{group}` | {row.get('line')} | `{text}` |")
    lines.extend([
        "",
        "## Boundary",
        "",
        "- This summary may bank only `banked_cutlass_abi=true`.",
        "- `banked_grouped_moe_fp4_mma_poc`, `banked_kernel_speed`, and `banked_default_promotion` must remain false.",
        "- A PASS means the next step is a minimal numeric GEMM smoke using CUTLASS/CuTe native NVF4 + UE4M3, not a grouped-MoE speed claim.",
        "",
    ])
    return "\n".join(lines)


def _strict_ok(data: dict[str, Any]) -> bool:
    passes = data.get("passes") or {}
    return bool(
        passes.get("banked_cutlass_abi") is True
        and passes.get("banked_grouped_moe_fp4_mma_poc") is False
        and passes.get("banked_kernel_speed") is False
        and passes.get("banked_default_promotion") is False
        and data.get("decision") == "PASS_R5C_NVF4_UE4M3_CUTLASS_ABI"
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
