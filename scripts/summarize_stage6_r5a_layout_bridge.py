#!/usr/bin/env python3
"""Summarize R5-A per-16 layout bridge probe artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def summarize(data: dict[str, Any], result_path: Path) -> str:
    passes = data.get("passes") or {}
    lines = [
        "# Stage 6 R5-A Layout Bridge Summary",
        "",
        f"| Field | Value |",
        "|---|---|",
        f"| Result | `{result_path}` |",
        f"| Decision | `{data.get('decision', 'unknown')}` |",
        f"| Layout bridge banked | `{passes.get('banked_layout_bridge')}` |",
        f"| Grouped MoE POC banked | `{passes.get('banked_grouped_moe_fp4_mma_poc')}` |",
        f"| Kernel speed banked | `{passes.get('banked_kernel_speed')}` |",
        f"| Default promotion banked | `{passes.get('banked_default_promotion')}` |",
        f"| Fold-pair group32 supported | `{passes.get('fold_pair_group32_supported')}` |",
        f"| Current Lynn E4M3 scales zero-copy supported | `{passes.get('current_lynn_e4m3_scales_zero_copy_supported')}` |",
        "",
        "## Candidate Rows",
        "",
        "| Scale case | M | Candidate | rel_l2 | cosine | median ms | packed ratio | scale ratio |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for case in data.get("cases") or []:
        shape = case.get("shape") or {}
        for row in case.get("candidates") or []:
            metrics = row.get("metrics") or {}
            timing = row.get("timing_ms") or {}
            byte_info = row.get("bytes") or {}
            lines.append(
                "| "
                f"{case.get('scale_case')} | "
                f"{shape.get('M')} | "
                f"`{row.get('candidate')}` | "
                f"{_fmt(metrics.get('rel_l2'))} | "
                f"{_fmt(metrics.get('cosine'))} | "
                f"{_fmt(timing.get('median_ms'))} | "
                f"{_fmt(byte_info.get('packed_ratio_vs_original'))} | "
                f"{_fmt(byte_info.get('scale_ratio_vs_original'))} |"
            )
    lines.extend([
        "",
        "## Boundary",
        "",
        "- This summary may bank only `banked_layout_bridge=true`.",
        "- `banked_grouped_moe_fp4_mma_poc`, `banked_kernel_speed`, and `banked_default_promotion` must remain false.",
        "- If current Lynn E4M3 scales are not zero-copy supported, R5-B must use explicit repack or custom scale handling.",
        "",
    ])
    return "\n".join(lines)


def _strict_ok(data: dict[str, Any]) -> bool:
    passes = data.get("passes") or {}
    return bool(
        passes.get("banked_layout_bridge") is True
        and passes.get("banked_grouped_moe_fp4_mma_poc") is False
        and passes.get("banked_kernel_speed") is False
        and passes.get("banked_default_promotion") is False
        and str(data.get("decision", "")).startswith("PASS_R5A_LAYOUT_BRIDGE")
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result_json")
    ap.add_argument("--markdown-out", default="")
    ap.add_argument("--strict-exit", action="store_true")
    args = ap.parse_args()

    result_path = Path(args.result_json)
    data = _load(result_path)
    md = summarize(data, result_path)
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n", encoding="utf-8")
    print(md)
    if args.strict_exit and not _strict_ok(data):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
