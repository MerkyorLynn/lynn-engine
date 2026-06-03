#!/usr/bin/env python3
"""Summarize Stage 6 P2-O packed-prefill RC smoke artifacts.

The P2-O runner writes a JSON artifact. This helper turns that artifact into a
stable Markdown verdict so README/report updates do not depend on hand-reading a
large JSON blob.
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


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [_f(row.get(key), default=float("nan")) for row in rows]
    vals = [v for v in vals if v == v]
    return statistics.fmean(vals) if vals else None


def _fmt(x: float | None, unit: str = "") -> str:
    if x is None or x != x:
        return "n/a"
    return f"{x:.3f}{unit}"


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    if passes.get("all") is True:
        return "PASS", "functional + token-exact"
    if passes.get("functional_non_degenerate") and passes.get("text_prefix_200_match"):
        return "WARN", "functional/text-prefix pass, token-exact fail"
    if passes.get("functional_non_degenerate"):
        return "FAIL", "functional pass, exact/text gate fail"
    return "FAIL", "functional non-degenerate gate fail"


def summarize(data: dict[str, Any]) -> str:
    baseline = data.get("baseline") or []
    optin = data.get("optin_no_reload") or []
    comparisons = data.get("comparisons") or []
    memory = data.get("memory") or {}
    release = memory.get("release") or {}
    passes = data.get("passes") or {}
    verdict, reason = _verdict(data)

    base_prefill = _avg(baseline, "prefill_seconds")
    opt_prefill = _avg(optin, "prefill_seconds")
    base_tps = _avg(baseline, "decode_tps")
    opt_tps = _avg(optin, "decode_tps")
    prefill_ratio = None
    if base_prefill and opt_prefill:
        prefill_ratio = base_prefill / opt_prefill
    tps_ratio = None
    if base_tps and opt_tps:
        tps_ratio = opt_tps / base_tps

    token_exact = sum(1 for row in comparisons if row.get("token_exact"))
    text_exact = sum(1 for row in comparisons if row.get("text_prefix_200_match"))
    n = len(comparisons)
    base_degenerate = sum(1 for row in baseline if row.get("degenerate"))
    opt_degenerate = sum(1 for row in optin if row.get("degenerate"))

    lines = [
        f"# Stage 6 P2-O RC Smoke Summary",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Preset | `{data.get('preset', 'unknown')}` |",
        f"| Model | `{data.get('model', 'unknown')}` |",
        f"| Max new tokens | `{data.get('max_new', 'unknown')}` |",
        f"| Functional non-degenerate | `{passes.get('functional_non_degenerate')}` |",
        f"| Token-exact prompts | `{token_exact}/{n}` |",
        f"| Text-prefix prompts | `{text_exact}/{n}` |",
        f"| Baseline degenerate | `{base_degenerate}/{len(baseline)}` |",
        f"| Opt-in degenerate | `{opt_degenerate}/{len(optin)}` |",
        f"| Loaded memory | `{_fmt(_f(memory.get('loaded_gib'), default=float('nan')), ' GiB')}` |",
        f"| After release memory | `{_fmt(_f(memory.get('after_release_gib'), default=float('nan')), ' GiB')}` |",
        f"| Released memory | `{_fmt(_f(release.get('released_gib'), default=float('nan')), ' GiB')}` |",
        f"| Baseline prefill avg | `{_fmt(base_prefill, ' s')}` |",
        f"| Opt-in prefill avg | `{_fmt(opt_prefill, ' s')}` |",
        f"| Prefill speed ratio | `{_fmt(prefill_ratio, 'x')}` |",
        f"| Baseline decode TPS avg | `{_fmt(base_tps)}` |",
        f"| Opt-in decode TPS avg | `{_fmt(opt_tps)}` |",
        f"| Decode TPS ratio | `{_fmt(tps_ratio, 'x')}` |",
        "",
        "## Per Prompt",
        "",
        "| # | Token Exact | Prefix | Baseline IDs | Opt-in IDs |",
        "|---|---|---:|---|---|",
    ]
    for row in comparisons:
        lines.append(
            "| {idx} | {exact} | {prefix}/{prefix_n} | `{base}` | `{cand}` |".format(
                idx=row.get("index"),
                exact=row.get("token_exact"),
                prefix=row.get("token_prefix_match"),
                prefix_n=row.get("token_prefix_n"),
                base=row.get("baseline_ids"),
                cand=row.get("optin_ids"),
            )
        )
    notes = data.get("notes") or []
    if notes:
        lines.extend(["", "## Caveats", ""])
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", help="Path to P2-O result.json")
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
