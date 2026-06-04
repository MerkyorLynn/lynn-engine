#!/usr/bin/env python3
"""Summarize R5-B e8m0 repack bridge artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def summarize(data: dict[str, Any], result_path: Path) -> str:
    passes = data.get("passes") or {}
    lines = [
        "# Stage 6 R5-B E8M0 Repack Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Result | `{result_path}` |",
        f"| Decision | `{data.get('decision', 'unknown')}` |",
        f"| Repack numeric banked | `{passes.get('banked_repack_numeric')}` |",
        f"| Grouped MoE POC banked | `{passes.get('banked_grouped_moe_fp4_mma_poc')}` |",
        f"| Kernel speed banked | `{passes.get('banked_kernel_speed')}` |",
        f"| Default promotion banked | `{passes.get('banked_default_promotion')}` |",
        "",
        "## Candidate Rows",
        "",
        "| M | Candidate | rel_l2 | cosine | median ms | act value rel_l2 | weight value rel_l2 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for case in data.get("cases") or []:
        shape = case.get("shape") or {}
        for row in case.get("candidates") or []:
            metrics = row.get("metrics") or {}
            timing = row.get("timing_ms") or {}
            act = row.get("act_repack") or {}
            weight = row.get("weight_repack") or {}
            lines.append(
                "| "
                f"{shape.get('M')} | "
                f"`{row.get('candidate')}` | "
                f"{_fmt(metrics.get('rel_l2'))} | "
                f"{_fmt(metrics.get('cosine'))} | "
                f"{_fmt(timing.get('median_ms'))} | "
                f"{_fmt(act.get('value_rel_l2'))} | "
                f"{_fmt(weight.get('value_rel_l2'))} |"
            )
    lines.extend([
        "",
        "## Boundary",
        "",
        "- This summary may bank only `banked_repack_numeric=true`.",
        "- `banked_grouped_moe_fp4_mma_poc`, `banked_kernel_speed`, and `banked_default_promotion` must remain false.",
        "- A PASS means R5-C can build the first selected/grouped-MoE FP4-MMA POC using the repacked artifact contract.",
        "",
    ])
    return "\n".join(lines)


def _strict_ok(data: dict[str, Any]) -> bool:
    passes = data.get("passes") or {}
    return bool(
        passes.get("banked_repack_numeric") is True
        and passes.get("banked_grouped_moe_fp4_mma_poc") is False
        and passes.get("banked_kernel_speed") is False
        and passes.get("banked_default_promotion") is False
        and str(data.get("decision", "")).startswith("PASS_R5B_E8M0_REPACK")
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
