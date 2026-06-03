#!/usr/bin/env python3
"""Write a Stage 6 P3-E RC quality-battery report from an artifact directory."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

from summarize_stage6_p3e_rc_quality_battery import _verdict, summarize


def _read(path: Path, default: str = "") -> str:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return default


def _tail(path: Path, lines: int) -> str:
    text = _read(path)
    if not text:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def _artifact_table(artifact_dir: Path) -> str:
    names = [
        "expected_git_head.txt",
        "expected_provenance_manifest.txt",
        "git_head.txt",
        "git_status.txt",
        "head_check.txt",
        "nvidia_smi_before.txt",
        "nvidia_smi_after.txt",
        "provenance_manifest.txt",
        "docker_exit_code.txt",
        "run.log",
        "candidate_server.log",
        "mmlu_sample.jsonl",
        "mmlu_sample.summary.json",
        "gpqa_sample.jsonl",
        "gpqa_sample.summary.json",
        "result.json",
        "summary.md",
    ]
    rows = ["| File | Present |", "|---|---|"]
    for name in names:
        rows.append(f"| `{name}` | `{(artifact_dir / name).exists()}` |")
    return "\n".join(rows)


def _fmt_bool(value: Any) -> str:
    return "true" if value is True else "false" if value is False else str(value)


def write_report(artifact_dir: Path, *, report_date: str) -> str:
    result_path = artifact_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"missing result.json in {artifact_dir}")
    data = json.loads(result_path.read_text())
    summary_path = artifact_dir / "summary.md"
    summary = summary_path.read_text().strip() if summary_path.exists() else summarize(data).strip()
    verdict, reason = _verdict(data)
    passes = data.get("passes") or {}
    git_head = _read(artifact_dir / "git_head.txt", "unknown")
    expected_head = _read(artifact_dir / "expected_git_head.txt", "unknown")
    manifest = _read(artifact_dir / "provenance_manifest.txt", "missing")
    expected_manifest = _read(artifact_dir / "expected_provenance_manifest.txt", "missing")
    head_check = _read(artifact_dir / "head_check.txt", "missing")
    git_status = _read(artifact_dir / "git_status.txt", "")
    gpu_before = _read(artifact_dir / "nvidia_smi_before.txt", "missing")
    gpu_after = _read(artifact_dir / "nvidia_smi_after.txt", "missing")
    docker_exit = _read(artifact_dir / "docker_exit_code.txt", "missing")
    run_log_tail = _tail(artifact_dir / "run.log", 100)
    server_log_tail = _tail(artifact_dir / "candidate_server.log", 80)

    decision = (
        "Bank P3-E as RC quality smoke only. Keep default promotion closed."
        if verdict == "PASS"
        else "Do not bank P3-E; inspect the failed quality-battery gate."
    )

    lines = [
        "# Stage 6 Phase 3-E — RC quality-battery smoke",
        "",
        f"Date: {report_date}",
        "",
        f"Verdict: **{verdict}** ({reason}).",
        "",
        "P3-E runs a compact quality battery against the opt-in zero-shadow server",
        "path after P3-D. It includes MMLU/GPQA samples, structured JSON, tool-call,",
        "V8/V9-shaped prompt smoke, and a long-context needle smoke.",
        "",
        "**Boundary:** a PASS banks only RC quality smoke. It is not default",
        "promotion and not a full leaderboard/long-run quality claim.",
        "",
        "## Artifact",
        "",
        f"Artifact directory: `{artifact_dir}`",
        "",
        _artifact_table(artifact_dir),
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Expected HEAD | `{expected_head}` |",
        f"| Remote HEAD | `{git_head}` |",
        f"| Head check | `{head_check}` |",
        f"| Manifest matches | `{manifest == expected_manifest}` |",
        f"| Docker exit code | `{docker_exit}` |",
        f"| Git status dirty | `{bool(git_status)}` |",
        f"| Model | `{data.get('model', 'unknown')}` |",
        f"| Served name | `{data.get('served_name', 'unknown')}` |",
        f"| Max seq len | `{data.get('max_seq_len', 'unknown')}` |",
        "",
        "GPU before:",
        "",
        "```text",
        gpu_before,
        "```",
        "",
        "GPU after:",
        "",
        "```text",
        gpu_after,
        "```",
        "",
        "Provenance manifest:",
        "",
        "```text",
        manifest,
        "```",
        "",
        "## Gate Summary",
        "",
        summary,
        "",
        "## Hard Gates",
        "",
        "| Gate | Value |",
        "|---|---|",
        f"| P3-D predecessor pass | `{_fmt_bool(passes.get('p3d_pass'))}` |",
        f"| Server ready | `{_fmt_bool(passes.get('server_ready'))}` |",
        f"| Models surface | `{_fmt_bool(passes.get('models_surface'))}` |",
        f"| Release enabled | `{_fmt_bool(passes.get('release_enabled'))}` |",
        f"| Skip reload enabled | `{_fmt_bool(passes.get('skip_reload_enabled'))}` |",
        f"| Zero reload observed | `{_fmt_bool(passes.get('zero_reload_observed'))}` |",
        f"| Structured JSON | `{_fmt_bool(passes.get('structured_json'))}` |",
        f"| Tool call | `{_fmt_bool(passes.get('tool_call'))}` |",
        f"| V8/V9 prompt smoke | `{_fmt_bool(passes.get('v8_v9_prompt_smoke'))}` |",
        f"| Long-context needle | `{_fmt_bool(passes.get('longctx_needle'))}` |",
        f"| MMLU data available | `{_fmt_bool(passes.get('mmlu_available'))}` |",
        f"| GPQA data available | `{_fmt_bool(passes.get('gpqa_available'))}` |",
        f"| MMLU score | `{_fmt_bool(passes.get('mmlu_score'))}` |",
        f"| GPQA score | `{_fmt_bool(passes.get('gpqa_score'))}` |",
        f"| No runner error | `{_fmt_bool(passes.get('no_runner_error'))}` |",
        f"| Banked RC quality smoke | `{data.get('banked_rc_quality_smoke')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        f"| Banked full leaderboard quality | `{data.get('banked_full_leaderboard_quality')}` |",
        "",
        "## Decision",
        "",
        decision,
        "",
        "Do not treat this report as full MMLU/GPQA leaderboard or default-promotion evidence.",
    ]
    if run_log_tail:
        lines.extend(["", "## Run Log Tail", "", "```text", run_log_tail, "```"])
    if server_log_tail:
        lines.extend(["", "## Candidate Server Log Tail", "", "```text", server_log_tail, "```"])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact_dir", help="Pulled P3-E artifact directory")
    ap.add_argument("--report-out", required=True, help="Markdown report output path")
    ap.add_argument("--date", default=str(_dt.date.today()))
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    report = write_report(artifact_dir, report_date=args.date)
    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
