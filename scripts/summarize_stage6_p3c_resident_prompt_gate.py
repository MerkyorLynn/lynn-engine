#!/usr/bin/env python3
"""Summarize Stage 6 P3-C resident-prompt gate artifacts."""
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
    if data.get("schema") != "lynn-stage6-p3c-resident-prompt-gate-v1":
        return "FAIL", "schema mismatch"
    if data.get("banked_server_path") is not False or data.get("banked_rc_quality") is not False:
        return "FAIL", "promotion boundary violated"
    hard_gates = (
        "p3b_pass",
        "prompt_count",
        "functional_non_degenerate",
        "generated_token_exact",
        "release_meaningful",
        "memory_drop_meaningful",
        "reload_not_called",
    )
    for gate in hard_gates:
        if passes.get(gate) is not True:
            return "FAIL", f"{gate} gate fail"
    if data.get("verdict") != "PASS":
        return "FAIL", "top-level verdict is not PASS"
    if passes.get("all") is True:
        return "PASS", "resident-prompt no-reload gate passed"
    return "FAIL", "resident-prompt aggregate gate fail"


def summarize(data: dict[str, Any]) -> str:
    baseline = data.get("baseline") or []
    candidate = data.get("candidate_no_reload") or []
    comparisons = data.get("comparisons") or []
    memory = data.get("memory") or {}
    release = memory.get("release") or {}
    passes = data.get("passes") or {}
    verdict, reason = _verdict(data)

    base_prefill = _avg(baseline, "prefill_seconds")
    cand_prefill = _avg(candidate, "prefill_seconds")
    base_tps = _avg(baseline, "decode_tps")
    cand_tps = _avg(candidate, "decode_tps")
    prefill_ratio = base_prefill / cand_prefill if base_prefill and cand_prefill else None
    tps_ratio = cand_tps / base_tps if base_tps and cand_tps else None
    token_exact = sum(1 for row in comparisons if row.get("token_exact"))
    text_exact = sum(1 for row in comparisons if row.get("text_prefix_200_match"))
    n = len(comparisons)
    base_degenerate = sum(1 for row in baseline if row.get("degenerate"))
    cand_degenerate = sum(1 for row in candidate if row.get("degenerate"))

    lines = [
        "# Stage 6 P3-C Resident-Prompt Gate Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Preset | `{data.get('preset', 'unknown')}` |",
        f"| Model | `{data.get('model', 'unknown')}` |",
        f"| Max new tokens | `{data.get('max_new', 'unknown')}` |",
        f"| Banked server path | `{data.get('banked_server_path')}` |",
        f"| Banked RC quality | `{data.get('banked_rc_quality')}` |",
        f"| P3-B predecessor pass | `{passes.get('p3b_pass')}` |",
        f"| Functional non-degenerate | `{passes.get('functional_non_degenerate')}` |",
        f"| Prompt count gate | `{passes.get('prompt_count')}` |",
        f"| Generated-ID exact prompts | `{token_exact}/{n}` |",
        f"| Text-prefix prompts | `{text_exact}/{n}` |",
        f"| Release meaningful | `{passes.get('release_meaningful')}` |",
        f"| Memory drop meaningful | `{passes.get('memory_drop_meaningful')}` |",
        f"| Reload not called | `{passes.get('reload_not_called')}` |",
        f"| Baseline degenerate | `{base_degenerate}/{len(baseline)}` |",
        f"| Candidate degenerate | `{cand_degenerate}/{len(candidate)}` |",
        f"| Loaded memory | `{_fmt(_f(memory.get('loaded_gib'), default=float('nan')), ' GiB')}` |",
        f"| After release memory | `{_fmt(_f(memory.get('after_release_gib'), default=float('nan')), ' GiB')}` |",
        f"| Memory drop | `{_fmt(_f(memory.get('drop_gib'), default=float('nan')), ' GiB')}` |",
        f"| Released memory | `{_fmt(_f(release.get('released_gib'), default=float('nan')), ' GiB')}` |",
        f"| Baseline prefill avg | `{_fmt(base_prefill, ' s')}` |",
        f"| Candidate prefill avg | `{_fmt(cand_prefill, ' s')}` |",
        f"| Prefill speed ratio | `{_fmt(prefill_ratio, 'x')}` |",
        f"| Baseline decode TPS avg | `{_fmt(base_tps)}` |",
        f"| Candidate decode TPS avg | `{_fmt(cand_tps)}` |",
        f"| Decode TPS ratio | `{_fmt(tps_ratio, 'x')}` |",
        "",
        "## Per Prompt",
        "",
        "| # | Token Exact | Prefix | Baseline IDs | Candidate IDs |",
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
                cand=row.get("candidate_ids"),
            )
        )
    notes = data.get("notes") or []
    if notes:
        lines.extend(["", "## Caveats", ""])
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", help="Path to P3-C result.json")
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
