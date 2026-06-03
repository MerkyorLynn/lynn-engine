#!/usr/bin/env python3
"""Summarize Stage 6 P3-B selected-prefill gate artifacts."""
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


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    if data.get("banked_fused_kernel") or data.get("banked_server_path"):
        return "FAIL", "promotion boundary violated"
    if passes.get("predecessors_pass") is not True:
        return "FAIL", "predecessor evidence gate fail"
    if passes.get("numeric") is not True:
        return "FAIL", "numeric gate fail"
    if passes.get("final_stack_argmax_match") is not True:
        return "FAIL", "argmax gate fail"
    if passes.get("no_active_bf16_shadow") is not True:
        return "FAIL", "active BF16 expert shadow was not absent"
    if passes.get("reload_not_called") is not True:
        return "FAIL", "reload was called"
    if passes.get("speed_vs_p2n_reference") is not True:
        return "FAIL", "P3-B candidate slower than P2-N reference"
    if passes.get("all") is True:
        return "PASS", "selected-prefill composition gates passed"
    return "FAIL", "selected-prefill aggregate gate fail"


def summarize(data: dict[str, Any]) -> str:
    passes = data.get("passes") or {}
    bytes_ = data.get("bytes") or {}
    bench = data.get("bench") or {}
    rows = bench.get("rows") or []
    verdict, reason = _verdict(data)
    avg_bf16 = _avg(rows, "bf16_prefill_us")
    avg_p2n = _avg(rows, "p2n_reference_us")
    avg_p3b = _avg(rows, "p3b_selected_prefill_us")
    avg_vs_p2n = _avg(rows, "p3b_vs_p2n")
    avg_vs_bf16 = _avg(rows, "p3b_vs_bf16")

    lines = [
        "# Stage 6 P3-B Selected-Prefill Gate Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Schema | `{data.get('schema', 'unknown')}` |",
        f"| Model | `{data.get('model', 'unknown')}` |",
        f"| Layers | `{data.get('layers', 'unknown')}` |",
        f"| Layer types | `{data.get('layer_types', 'unknown')}` |",
        f"| Seq lens | `{data.get('seq_lens', 'unknown')}` |",
        f"| Banked fused kernel | `{data.get('banked_fused_kernel')}` |",
        f"| Banked server path | `{data.get('banked_server_path')}` |",
        f"| Predecessors pass | `{passes.get('predecessors_pass')}` |",
        f"| Numeric | `{passes.get('numeric')}` |",
        f"| Final stack cosine min | `{_fmt(_f(passes.get('final_stack_cosine_min'), default=float('nan')))}` |",
        f"| Final stack argmax | `{passes.get('final_stack_argmax_match')}` |",
        f"| Active BF16 shadow absent | `{passes.get('no_active_bf16_shadow')}` |",
        f"| Reload not called | `{passes.get('reload_not_called')}` |",
        f"| Speed vs P2-N reference | `{passes.get('speed_vs_p2n_reference')}` |",
        f"| BF16 active expert bytes | `{bytes_.get('bf16_active_experts', 'unknown')}` |",
        f"| Packed active expert bytes | `{bytes_.get('packed_active_experts', 'unknown')}` |",
        f"| Memory after load | `{_fmt(_f(bytes_.get('mem_after_load_gib'), default=float('nan')), ' GiB')}` |",
        f"| Memory after active-shadow delete | `{_fmt(_f(bytes_.get('mem_after_deleting_bf16_active_gib'), default=float('nan')), ' GiB')}` |",
        f"| Active-shadow memory drop | `{_fmt(_f(bytes_.get('mem_drop_after_deleting_bf16_active_gib'), default=float('nan')), ' GiB')}` |",
        f"| Avg BF16 prefill | `{_fmt(avg_bf16, ' us')}` |",
        f"| Avg P2-N reference | `{_fmt(avg_p2n, ' us')}` |",
        f"| Avg P3-B candidate | `{_fmt(avg_p3b, ' us')}` |",
        f"| Avg P3-B vs BF16 | `{_fmt(avg_vs_bf16, 'x')}` |",
        f"| Avg P3-B vs P2-N | `{_fmt(avg_vs_p2n, 'x')}` |",
        "",
        "## Per Sequence",
        "",
        "| Seq | BF16 us | P2-N us | P3-B us | P3-B/BF16 | P3-B/P2-N | Cosine | Argmax |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {seq} | {bf16} | {p2n} | {p3b} | {vs_bf16} | {vs_p2n} | {cos} | `{argmax}` |".format(
                seq=row.get("seq_len"),
                bf16=_fmt(_f(row.get("bf16_prefill_us"), default=float("nan"))),
                p2n=_fmt(_f(row.get("p2n_reference_us"), default=float("nan"))),
                p3b=_fmt(_f(row.get("p3b_selected_prefill_us"), default=float("nan"))),
                vs_bf16=_fmt(_f(row.get("p3b_vs_bf16"), default=float("nan")), "x"),
                vs_p2n=_fmt(_f(row.get("p3b_vs_p2n"), default=float("nan")), "x"),
                cos=_fmt(_f(row.get("p3b_cosine_vs_bf16"), default=float("nan"))),
                argmax=row.get("p3b_argmax_vs_bf16"),
            )
        )
    notes = data.get("notes") or []
    if notes:
        lines.extend(["", "## Caveats", ""])
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", help="Path to P3-B result.json")
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
