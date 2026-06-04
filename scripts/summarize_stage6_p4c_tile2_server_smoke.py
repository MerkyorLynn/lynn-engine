#!/usr/bin/env python3
"""Summarize Stage 6 P4C tile=2 server smoke artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "lynn-stage6-p4c-tile2-server-smoke-v1"
EXPECTED_BACKEND = "fused_zero_shadow_active_reuse_contract"


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    if data.get("schema") != SCHEMA:
        return "FAIL", "schema mismatch"
    if data.get("banked_default_promotion") is not False:
        return "FAIL", "default promotion boundary violated"
    if data.get("banked_full_rc_quality") is not False:
        return "FAIL", "full RC quality boundary violated"
    if ((data.get("env") or {}).get("candidate") or {}).get("LYNN_NATIVE_ACTIVE_MOE_BACKEND") != EXPECTED_BACKEND:
        return "FAIL", "candidate backend mismatch"
    hard_gates = (
        "p4c_runtime_predecessor_pass",
        "server_surface",
        "prompt_count",
        "functional_non_degenerate",
        "server_text_exact",
        "candidate_runtime_env",
        "p4c_native_backend_called",
        "p4c_tile_recorded",
        "p4c_active_reuse_shapes_recorded",
        "release_enabled",
        "release_consumed",
        "decode_shadows_currently_released",
        "release_meaningful",
        "reload_observed",
        "default_promotion_closed",
        "full_rc_quality_unbanked",
    )
    for gate in hard_gates:
        if passes.get(gate) is not True:
            return "FAIL", f"{gate} gate fail"
    if data.get("banked_p4c_tile2_server_smoke") is not True:
        return "FAIL", "server smoke was not banked"
    if data.get("verdict") != "PASS":
        return "FAIL", "top-level verdict is not PASS"
    if passes.get("all") is True:
        return "PASS", "P4C tile2 server smoke passed"
    return "FAIL", "aggregate gate fail"


def _count_exact(rows: list[dict[str, Any]]) -> tuple[int, int]:
    return sum(1 for row in rows if row.get("text_exact")), len(rows)


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    passes = data.get("passes") or {}
    comparisons = data.get("comparisons") or {}
    completion_exact, completion_n = _count_exact(comparisons.get("completions") or [])
    chat_exact, chat_n = _count_exact(comparisons.get("chat") or [])
    baseline = data.get("baseline") or {}
    candidate = data.get("candidate") or {}
    baseline_rows = (baseline.get("completions") or []) + (baseline.get("chat") or [])
    candidate_rows = (candidate.get("completions") or []) + (candidate.get("chat") or [])
    baseline_degenerate = sum(1 for row in baseline_rows if row.get("degenerate"))
    candidate_degenerate = sum(1 for row in candidate_rows if row.get("degenerate"))
    native = data.get("candidate_native_counter") or {}
    health = data.get("candidate_health") or {}
    shapes = native.get("last_shapes") or {}

    lines = [
        "# Stage 6 P4C Tile2 Server Smoke Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Decision | `{data.get('decision')}` |",
        f"| Model | `{data.get('model', 'unknown')}` |",
        f"| Preset | `{data.get('preset', 'unknown')}` |",
        f"| Prompt limit | `{data.get('prompt_limit', 'unknown')}` |",
        f"| Max new tokens | `{data.get('max_new', 'unknown')}` |",
        f"| Candidate backend | `{((data.get('env') or {}).get('candidate') or {}).get('LYNN_NATIVE_ACTIVE_MOE_BACKEND')}` |",
        f"| Gate/up tile_inter | `{data.get('gateup_tile_inter')}` |",
        f"| Banked P4C server smoke | `{data.get('banked_p4c_tile2_server_smoke')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        f"| Banked full RC quality | `{data.get('banked_full_rc_quality')}` |",
        f"| P4C runtime predecessor pass | `{passes.get('p4c_runtime_predecessor_pass')}` |",
        f"| Server surface | `{passes.get('server_surface')}` |",
        f"| Functional non-degenerate | `{passes.get('functional_non_degenerate')}` |",
        f"| Completion text exact | `{completion_exact}/{completion_n}` |",
        f"| Chat text exact | `{chat_exact}/{chat_n}` |",
        f"| Baseline degenerate | `{baseline_degenerate}/{len(baseline_rows)}` |",
        f"| Candidate degenerate | `{candidate_degenerate}/{len(candidate_rows)}` |",
        f"| P4C native call delta | `{native.get('delta_total_calls')}` |",
        f"| P4C native calls after | `{native.get('after_total_calls')}` |",
        f"| P4C layers with calls | `{(native.get('after') or {}).get('layers_with_calls')}` |",
        f"| Recorded tile_inter | `{shapes.get('gateup_tile_inter')}` |",
        f"| Recorded inter scratch | `{shapes.get('inter_scratch')}` |",
        f"| Recorded out scratch | `{shapes.get('out')}` |",
        f"| Release enabled | `{passes.get('release_enabled')}` |",
        f"| Release consumed | `{passes.get('release_consumed')}` |",
        f"| Decode shadows currently released | `{passes.get('decode_shadows_currently_released')}` |",
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
    ap.add_argument("result_json", help="Path to P4C tile2 server smoke result.json")
    ap.add_argument("--markdown-out", default="", help="Optional Markdown output path")
    ap.add_argument("--strict-exit", action="store_true", help="Exit non-zero unless verdict is PASS")
    args = ap.parse_args()

    data = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    md = summarize(data)
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
    sys.stdout.write(md)
    verdict, _ = _verdict(data)
    return 0 if (verdict == "PASS" or not args.strict_exit) else 2


if __name__ == "__main__":
    raise SystemExit(main())
