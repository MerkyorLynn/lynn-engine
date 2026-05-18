#!/usr/bin/env python3
"""Summarize Qwen3.6 W4A16 promotion gate reports.

The promotion wrapper intentionally emits one JSON file per candidate run. This
utility keeps the coordination surface small when several agents are producing
candidate reports in parallel.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _short_decision(decision: str) -> str:
    return decision.split(":", 1)[0] if decision else "UNKNOWN"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_bool(value: Any) -> str:
    if value is True:
        return "Y"
    if value is False:
        return "N"
    return "-"


def _row(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics") or {}
    thresholds = report.get("thresholds") or {}
    return {
        "path": str(path),
        "candidate": report.get("candidate_name") or path.name,
        "stamp": report.get("stamp"),
        "decision": report.get("decision") or "",
        "decision_class": _short_decision(report.get("decision") or ""),
        "p37_exact": metrics.get("p37_exact"),
        "p37_median_speedup": metrics.get("p37_median_speedup"),
        "p25_512_decode_tps": metrics.get("p25_512_decode_tps"),
        "structured_ok": metrics.get("hard_structured_ok"),
        "structured_decode_tps_mean": metrics.get("hard_structured_decode_tps_mean"),
        "structured_request_count": metrics.get("structured_request_count"),
        "structured_pass_count": metrics.get("structured_pass_count"),
        "structured_40_exact": metrics.get("structured_40_exact"),
        "structured_70_exact": metrics.get("structured_70_exact"),
        "default_threshold": thresholds.get("default_threshold"),
        "amber_threshold": thresholds.get("amber_threshold"),
        "candidate_env": report.get("candidate_env") or [],
    }


def _sort_key(row: dict[str, Any]) -> tuple[int, float, str]:
    rank = {
        "DEFAULT_CANDIDATE": 0,
        "AMBER_CANDIDATE": 1,
        "RESEARCH_ONLY": 2,
        "CLOSED": 3,
    }.get(row["decision_class"], 4)
    tps = row.get("p25_512_decode_tps")
    try:
        score = -float(tps)
    except (TypeError, ValueError):
        score = 0.0
    return rank, score, str(row.get("candidate"))


def _markdown(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Candidate | Decision | P37 | P25 512 TPS | Structured | "
        "Struct TPS | Thresholds | Env |\n"
        "|---|---:|---:|---:|---:|---:|---:|---|\n"
    )
    lines = []
    for row in rows:
        req = row.get("structured_request_count")
        passed = row.get("structured_pass_count")
        structured = "-"
        if req is not None and passed is not None:
            structured = f"{passed}/{req}"
        elif row.get("structured_ok") is not None:
            structured = _fmt_bool(row.get("structured_ok"))
        thresholds = f"{_fmt_float(row.get('default_threshold'))}/{_fmt_float(row.get('amber_threshold'))}"
        env = "<br>".join(row.get("candidate_env") or ["safe-default"])
        lines.append(
            "| {candidate} | {decision} | {p37} | {p25} | {structured} | "
            "{struct_tps} | {thresholds} | {env} |".format(
                candidate=row["candidate"],
                decision=row["decision_class"],
                p37=_fmt_bool(row.get("p37_exact")),
                p25=_fmt_float(row.get("p25_512_decode_tps")),
                structured=structured,
                struct_tps=_fmt_float(row.get("structured_decode_tps_mean")),
                thresholds=thresholds,
                env=env,
            )
        )
    return header + "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        default=["reports/qwen36_35b/*promotion_summary.json"],
        help="Promotion summary JSON paths or globs.",
    )
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    paths: list[Path] = []
    for item in args.paths:
        matches = glob.glob(item)
        paths.extend(Path(match) for match in matches) if matches else paths.append(Path(item))

    rows = []
    for path in sorted(set(paths)):
        report = _load(path)
        if report is not None and report.get("schema_version") == "lynn-qwen36-candidate-promotion-summary-v1":
            rows.append(_row(path, report))
    rows.sort(key=_sort_key)

    output = json.dumps(rows, ensure_ascii=False, indent=2) + "\n" if args.format == "json" else _markdown(rows)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
