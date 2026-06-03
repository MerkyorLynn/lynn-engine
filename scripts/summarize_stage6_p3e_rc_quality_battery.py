#!/usr/bin/env python3
"""Summarize Stage 6 P3-E RC quality-battery smoke artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _summary(data: dict[str, Any], task: str) -> dict[str, Any]:
    return (((data.get("mcq") or {}).get(task) or {}).get("summary_data") or {})


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    if data.get("schema") != "lynn-stage6-p3e-rc-quality-battery-v1":
        return "FAIL", "schema mismatch"
    if data.get("banked_default_promotion") is not False:
        return "FAIL", "default promotion boundary violated"
    if data.get("banked_full_leaderboard_quality") is not False:
        return "FAIL", "full leaderboard boundary violated"
    hard_gates = (
        "p3d_pass",
        "server_ready",
        "models_surface",
        "release_enabled",
        "skip_reload_enabled",
        "zero_reload_observed",
        "structured_json",
        "tool_call",
        "v8_v9_prompt_smoke",
        "longctx_needle",
        "mmlu_available",
        "gpqa_available",
        "mmlu_score",
        "gpqa_score",
        "no_runner_error",
    )
    for gate in hard_gates:
        if passes.get(gate) is not True:
            return "FAIL", f"{gate} gate fail"
    if data.get("banked_rc_quality_smoke") is not True:
        return "FAIL", "RC quality smoke was not banked"
    if data.get("verdict") != "PASS":
        return "FAIL", "top-level verdict is not PASS"
    if passes.get("all") is True:
        return "PASS", "RC quality-battery smoke passed"
    return "FAIL", "aggregate gate fail"


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    passes = data.get("passes") or {}
    thresholds = data.get("thresholds") or {}
    mmlu = _summary(data, "mmlu")
    gpqa = _summary(data, "gpqa")
    health = data.get("health_after") or data.get("health_before") or {}
    prompt_rows = ((data.get("prompt_smoke") or {}).get("rows") or [])

    lines = [
        "# Stage 6 P3-E RC Quality-Battery Smoke Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Model | `{data.get('model', 'unknown')}` |",
        f"| Served name | `{data.get('served_name', 'unknown')}` |",
        f"| Max seq len | `{data.get('max_seq_len', 'unknown')}` |",
        f"| Banked RC quality smoke | `{data.get('banked_rc_quality_smoke')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        f"| Banked full leaderboard quality | `{data.get('banked_full_leaderboard_quality')}` |",
        f"| P3-D predecessor pass | `{passes.get('p3d_pass')}` |",
        f"| Server ready | `{passes.get('server_ready')}` |",
        f"| Models surface | `{passes.get('models_surface')}` |",
        f"| Release enabled | `{passes.get('release_enabled')}` |",
        f"| Skip reload enabled | `{passes.get('skip_reload_enabled')}` |",
        f"| Zero reload observed | `{passes.get('zero_reload_observed')}` |",
        f"| Release/reload count | `{health.get('release_reload_count')}` |",
        f"| Last release GiB | `{health.get('last_release_gib')}` |",
        f"| Structured JSON | `{passes.get('structured_json')}` |",
        f"| Tool call | `{passes.get('tool_call')}` |",
        f"| V8/V9 smoke | `{passes.get('v8_v9_prompt_smoke')}` |",
        f"| Long-context needle | `{passes.get('longctx_needle')}` |",
        f"| MMLU sample | `{mmlu.get('correct')}/{mmlu.get('n')}` acc `{_f(mmlu.get('accuracy')):.4f}` floor `{thresholds.get('mmlu_floor')}` |",
        f"| MMLU parse/errors | `{mmlu.get('parse_fail')}` / `{mmlu.get('errors')}` |",
        f"| GPQA sample | `{gpqa.get('correct')}/{gpqa.get('n')}` acc `{_f(gpqa.get('accuracy')):.4f}` floor `{thresholds.get('gpqa_floor')}` |",
        f"| GPQA parse/errors | `{gpqa.get('parse_fail')}` / `{gpqa.get('errors')}` |",
        f"| Wall seconds | `{data.get('wall_seconds')}` |",
        "",
        "## Prompt-Format Smoke",
        "",
        "| Smoke | OK | Chars |",
        "|---|---:|---:|",
    ]
    for row in prompt_rows:
        lines.append(f"| `{row.get('id')}` | `{row.get('ok')}` | `{len(str(row.get('text') or ''))}` |")
    notes = data.get("notes") or []
    if notes:
        lines.extend(["", "## Caveats", ""])
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", help="Path to P3-E result.json")
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
