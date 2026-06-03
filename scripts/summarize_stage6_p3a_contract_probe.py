#!/usr/bin/env python3
"""Summarize Stage 6 P3-A grouped-MoE contract probe artifacts.

The P3-A runner emits a JSON artifact. This helper turns it into a stable
Markdown verdict and keeps the promotion boundary explicit: P3-A can pass the
contract probe, but it must not be reported as a banked fused kernel.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _fmt(x: float | None, unit: str = "") -> str:
    if x is None or x != x:
        return "n/a"
    return f"{x:.3f}{unit}"


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [_f(row.get(key), default=float("nan")) for row in rows]
    vals = [v for v in vals if v == v]
    return statistics.fmean(vals) if vals else None


def _bytes_to_gib(n: Any) -> float | None:
    value = _f(n, default=float("nan"))
    if value != value:
        return None
    return value / 1024**3


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    if data.get("banked_fused_kernel") is not False:
        return "FAIL", "promotion boundary violated: banked_fused_kernel must be false"
    if passes.get("shadow_absent_at_candidate_start") is not True:
        return "FAIL", "BF16 active shadow was not absent at candidate start"
    if passes.get("numeric") is not True:
        return "FAIL", "numeric gate fail"
    if passes.get("all") is True:
        return "PASS", "P3-A contract probe pass; fused kernel not banked"
    return "FAIL", "aggregate pass flag false"


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    passes = data.get("passes") or {}
    rows = ((data.get("bench") or {}).get("rows") or [])
    shape = data.get("shape") or {}
    tiles = data.get("tiles") or {}
    bytes_ = data.get("bytes") or {}
    memory = data.get("memory") or {}
    peak = memory.get("p3a_candidate_peak") or {}

    avg_speed = _avg(rows, "p3a_vs_bf16")
    min_cos = min((_f(row.get("cosine"), default=float("nan")) for row in rows), default=float("nan"))
    argmax_matches = sum(1 for row in rows if row.get("argmax_match") is True)
    peak_vals = [
        _f(item.get("peak_gib"), default=float("nan"))
        for item in peak.values()
        if isinstance(item, dict)
    ]
    peak_vals = [v for v in peak_vals if v == v]
    max_peak = max(peak_vals) if peak_vals else None

    lines = [
        "# Stage 6 P3-A Grouped-MoE Contract Probe Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Banked fused kernel | `{data.get('banked_fused_kernel')}` |",
        f"| Model | `{data.get('model', 'unknown')}` |",
        f"| Layer | `{data.get('layer', 'unknown')}` |",
        f"| Batches | `{data.get('batches', [])}` |",
        f"| Shape | `H={shape.get('hidden')} I={shape.get('intermediate')} E={shape.get('num_experts')} top_k={shape.get('top_k')}` |",
        f"| Tiles | `gate T={tiles.get('block_t')} I={tiles.get('block_inter')} H={tiles.get('block_hidden')}; down H={tiles.get('down_block_hidden')} I={tiles.get('down_block_inter')}` |",
        f"| Numeric gate | `{passes.get('numeric')}` |",
        f"| Shadow absent at candidate start | `{passes.get('shadow_absent_at_candidate_start')}` |",
        f"| Aggregate pass | `{passes.get('all')}` |",
        f"| BF16 active expert bytes | `{_fmt(_bytes_to_gib(bytes_.get('bf16_layer_active_experts')), ' GiB')}` |",
        f"| Packed active expert bytes | `{_fmt(_bytes_to_gib(bytes_.get('packed_layer_active_experts')), ' GiB')}` |",
        f"| Inter scratch estimate | `{_fmt(_bytes_to_gib(bytes_.get('max_inter_scratch_estimate')), ' GiB')}` |",
        f"| Memory after deleting BF16 active | `{_fmt(_f(bytes_.get('mem_after_deleting_bf16_active_gib'), default=float('nan')), ' GiB')}` |",
        f"| Max candidate peak | `{_fmt(max_peak, ' GiB')}` |",
        f"| Average P3-A vs BF16 speed | `{_fmt(avg_speed, 'x')}` |",
        f"| Min cosine | `{_fmt(min_cos)}` |",
        f"| Argmax matches | `{argmax_matches}/{len(rows)}` |",
        "",
        "## Per Batch",
        "",
        "| Batch | Unique experts | BF16 active us | P3-A us | Speed | Cosine | Argmax |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {batch} | {experts} | {bf16} | {p3a} | {speed} | {cos} | {argmax} |".format(
                batch=row.get("batch"),
                experts=row.get("unique_experts"),
                bf16=_fmt(_f(row.get("bf16_active_us"), default=float("nan"))),
                p3a=_fmt(_f(row.get("p3a_contract_us"), default=float("nan"))),
                speed=_fmt(_f(row.get("p3a_vs_bf16"), default=float("nan")), "x"),
                cos=_fmt(_f(row.get("cosine"), default=float("nan"))),
                argmax=row.get("argmax_match"),
            )
        )
    notes = data.get("notes") or []
    if notes:
        lines.extend(["", "## Caveats", ""])
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", help="Path to P3-A result.json")
    ap.add_argument("--markdown-out", default="", help="Optional Markdown output path")
    ap.add_argument("--strict-exit", action="store_true", help="Exit non-zero unless verdict is PASS")
    args = ap.parse_args()

    data = json.loads(Path(args.result_json).read_text())
    md = summarize(data)
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
    sys.stdout.write(md)
    verdict, _ = _verdict(data)
    return 0 if (verdict == "PASS" or not args.strict_exit) else 2


if __name__ == "__main__":
    raise SystemExit(main())
