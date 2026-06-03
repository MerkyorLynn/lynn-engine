#!/usr/bin/env python3
"""Summarize Stage 6 P3-D server RC smoke artifacts."""
from __future__ import annotations

import argparse
import json
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


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    if data.get("schema") != "lynn-stage6-p3d-server-rc-gate-v1":
        return "FAIL", "schema mismatch"
    if data.get("banked_default_promotion") is not False:
        return "FAIL", "default promotion boundary violated"
    if data.get("banked_full_rc_quality") is not False:
        return "FAIL", "full RC quality boundary violated"
    hard_gates = (
        "p3c_pass",
        "server_surface",
        "prompt_count",
        "functional_non_degenerate",
        "server_text_exact",
        "release_enabled",
        "release_consumed",
        "decode_shadows_currently_released",
        "release_meaningful",
        "reload_observed",
    )
    for gate in hard_gates:
        if passes.get(gate) is not True:
            return "FAIL", f"{gate} gate fail"
    if data.get("banked_server_smoke") is not True:
        return "FAIL", "server smoke was not banked"
    if data.get("verdict") != "PASS":
        return "FAIL", "top-level verdict is not PASS"
    if passes.get("all") is True:
        return "PASS", "server smoke gate passed"
    return "FAIL", "aggregate gate fail"


def _count_exact(rows: list[dict[str, Any]]) -> tuple[int, int]:
    return sum(1 for row in rows if row.get("text_exact")), len(rows)


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    passes = data.get("passes") or {}
    health = data.get("candidate_health") or {}
    comparisons = data.get("comparisons") or {}
    completion_exact, completion_n = _count_exact(comparisons.get("completions") or [])
    chat_exact, chat_n = _count_exact(comparisons.get("chat") or [])
    baseline = data.get("baseline") or {}
    candidate = data.get("candidate") or {}
    baseline_rows = (baseline.get("completions") or []) + (baseline.get("chat") or [])
    candidate_rows = (candidate.get("completions") or []) + (candidate.get("chat") or [])
    baseline_degenerate = sum(1 for row in baseline_rows if row.get("degenerate"))
    candidate_degenerate = sum(1 for row in candidate_rows if row.get("degenerate"))

    lines = [
        "# Stage 6 P3-D Server RC Smoke Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Preset | `{data.get('preset', 'unknown')}` |",
        f"| Model | `{data.get('model', 'unknown')}` |",
        f"| Max new tokens | `{data.get('max_new', 'unknown')}` |",
        f"| Banked server smoke | `{data.get('banked_server_smoke')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        f"| Banked full RC quality | `{data.get('banked_full_rc_quality')}` |",
        f"| P3-C predecessor pass | `{passes.get('p3c_pass')}` |",
        f"| Server surface | `{passes.get('server_surface')}` |",
        f"| Prompt count | `{passes.get('prompt_count')}` |",
        f"| Functional non-degenerate | `{passes.get('functional_non_degenerate')}` |",
        f"| Completion text exact | `{completion_exact}/{completion_n}` |",
        f"| Chat text exact | `{chat_exact}/{chat_n}` |",
        f"| Baseline degenerate | `{baseline_degenerate}/{len(baseline_rows)}` |",
        f"| Candidate degenerate | `{candidate_degenerate}/{len(candidate_rows)}` |",
        f"| Release enabled | `{passes.get('release_enabled')}` |",
        f"| Release consumed | `{passes.get('release_consumed')}` |",
        f"| Decode shadows currently released | `{passes.get('decode_shadows_currently_released')}` |",
        f"| Release meaningful | `{passes.get('release_meaningful')}` |",
        f"| Reload observed | `{passes.get('reload_observed')}` |",
        f"| Release/reload count | `{health.get('release_reload_count')}` |",
        f"| Reload expected min | `{health.get('reload_expected_min')}` |",
        f"| Last release GiB | `{_f(health.get('last_release_gib')):.3f}` |",
        f"| Last reload seconds | `{health.get('last_reload_seconds')}` |",
        "",
        "## Completion Comparisons",
        "",
        "| # | Text Exact | Baseline | Candidate |",
        "|---|---|---|---|",
    ]
    for row in comparisons.get("completions") or []:
        lines.append(
            "| {idx} | {exact} | `{base}` | `{cand}` |".format(
                idx=row.get("index"),
                exact=row.get("text_exact"),
                base=str(row.get("baseline_text", ""))[:120].replace("`", "'"),
                cand=str(row.get("candidate_text", ""))[:120].replace("`", "'"),
            )
        )
    if comparisons.get("chat"):
        lines.extend(["", "## Chat Comparisons", "", "| # | Text Exact | Baseline | Candidate |", "|---|---|---|---|"])
        for row in comparisons.get("chat") or []:
            lines.append(
                "| {idx} | {exact} | `{base}` | `{cand}` |".format(
                    idx=row.get("index"),
                    exact=row.get("text_exact"),
                    base=str(row.get("baseline_text", ""))[:120].replace("`", "'"),
                    cand=str(row.get("candidate_text", ""))[:120].replace("`", "'"),
                )
            )
    notes = data.get("notes") or []
    if notes:
        lines.extend(["", "## Caveats", ""])
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", help="Path to P3-D result.json")
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
