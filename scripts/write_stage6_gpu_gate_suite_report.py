#!/usr/bin/env python3
"""Write a formal Stage 6 GPU gate-suite report from a suite artifact dir."""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path


def _read(path: Path, default: str = "") -> str:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return default


def _parse_meta(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key] = value
    return out


def _parse_status(path: Path) -> list[dict[str, str]]:
    text = _read(path)
    rows: list[dict[str, str]] = []
    for idx, line in enumerate(text.splitlines()):
        if idx == 0:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        rows.append({"step": parts[0], "status": parts[1], "exit_code": parts[2]})
    return rows


def _child_dirs(suite_dir: Path) -> list[Path]:
    prefixes = (
        "p2o_basic_packed_prefill_rc_smoke_",
        "p2o_rc-mini_packed_prefill_rc_smoke_",
        "p3a_layer",
    )
    return sorted(
        p
        for p in suite_dir.iterdir()
        if p.is_dir() and any(p.name.startswith(prefix) for prefix in prefixes)
    )


def _child_table(children: list[Path]) -> str:
    rows = [
        "| Child artifact | result.json | summary.md | head_check | docker exit |",
        "|---|---:|---:|---|---:|",
    ]
    for child in children:
        rows.append(
            "| `{name}` | `{result}` | `{summary}` | `{head}` | `{docker}` |".format(
                name=child.name,
                result=(child / "result.json").exists(),
                summary=(child / "summary.md").exists(),
                head=_read(child / "head_check.txt", "missing").replace("\n", " / "),
                docker=_read(child / "docker_exit_code.txt", "missing"),
            )
        )
    if not children:
        rows.append("| _none_ | `False` | `False` | `n/a` | `n/a` |")
    return "\n".join(rows)


def _embed_child_summaries(children: list[Path]) -> list[str]:
    lines: list[str] = []
    for child in children:
        summary = _read(child / "summary.md")
        if not summary:
            continue
        lines.extend([
            f"### `{child.name}`",
            "",
            summary,
            "",
        ])
    if not lines:
        lines.extend(["_No child summaries present._", ""])
    return lines


def write_report(suite_dir: Path, *, report_date: str) -> str:
    meta = _parse_meta(_read(suite_dir / "suite_meta.env"))
    statuses = _parse_status(suite_dir / "suite_status.tsv")
    commands = _read(suite_dir / "commands.sh")
    local_git_status = _read(suite_dir / "local_git_status.txt")
    suite_summary = _read(suite_dir / "summary.md")
    children = _child_dirs(suite_dir)
    fail_count = sum(1 for row in statuses if row["status"] == "FAIL")
    dry_count = sum(1 for row in statuses if row["status"] == "DRY_RUN")
    pass_count = sum(1 for row in statuses if row["status"] == "PASS")
    skip_count = sum(1 for row in statuses if row["status"] == "SKIP")
    verdict = "PASS" if fail_count == 0 and dry_count == 0 else "FAIL" if fail_count else "DRY_RUN"
    reason = (
        "all executed child gates passed"
        if verdict == "PASS"
        else "one or more child gates failed"
        if verdict == "FAIL"
        else "dry-run only; no GPU gate was executed"
    )
    decision = (
        "Use child P2-O/P3-A reports to decide what can be banked; this suite report only aggregates evidence."
        if verdict == "PASS"
        else "Do not bank new GPU results from this suite."
    )

    lines = [
        "# Stage 6 GPU Gate Suite Report",
        "",
        f"Date: {report_date}",
        "",
        f"Verdict: **{verdict}** ({reason}).",
        "",
        "This suite report aggregates the current Stage 6 GPU gates. It does not",
        "promote P2-O or P3 by itself; child reports remain authoritative.",
        "",
        "## Suite Artifact",
        "",
        f"Suite directory: `{suite_dir}`",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Local HEAD | `{meta.get('local_head', 'unknown')}` |",
        f"| Expected Spark HEAD | `{meta.get('expected_head', 'unknown')}` |",
        f"| Host | `{meta.get('host', 'unknown')}` |",
        f"| Model | `{meta.get('model', 'unknown')}` |",
        f"| Image | `{meta.get('image', 'unknown')}` |",
        f"| Remote repo | `{meta.get('remote_repo', 'unknown')}` |",
        f"| Strict | `{meta.get('strict', 'unknown')}` |",
        f"| Dry run | `{meta.get('dry_run', 'unknown')}` |",
        f"| Pass / Fail / Dry / Skip | `{pass_count} / {fail_count} / {dry_count} / {skip_count}` |",
        f"| Local git status dirty | `{bool(local_git_status)}` |",
        "",
        "## Step Status",
        "",
        "| Step | Status | Exit code |",
        "|---|---|---:|",
    ]
    for row in statuses:
        lines.append(f"| `{row['step']}` | `{row['status']}` | `{row['exit_code']}` |")
    if not statuses:
        lines.append("| _none_ | `missing` | `n/a` |")
    lines.extend([
        "",
        "## Child Artifacts",
        "",
        _child_table(children),
        "",
        "## Suite Summary",
        "",
        suite_summary or "_missing suite summary_",
        "",
        "## Child Summaries",
        "",
    ])
    lines.extend(_embed_child_summaries(children))
    lines.extend([
        "## Commands",
        "",
        "```bash",
        commands,
        "```",
        "",
        "## Decision",
        "",
        decision,
        "",
        "A PASS here is orchestration-level evidence only. P2-O and P3-A must still",
        "be banked through their own report writers and gate-specific caveats.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite_dir", help="Stage 6 GPU gate-suite artifact directory")
    ap.add_argument("--report-out", required=True, help="Markdown report output path")
    ap.add_argument("--date", default=str(_dt.date.today()))
    args = ap.parse_args()

    suite_dir = Path(args.suite_dir)
    report = write_report(suite_dir, report_date=args.date)
    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
