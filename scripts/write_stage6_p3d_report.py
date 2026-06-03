#!/usr/bin/env python3
"""Write a Stage 6 P3-D server smoke report from an artifact directory."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

from summarize_stage6_p3d_server_rc_gate import _verdict, summarize


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
        "baseline_server.log",
        "candidate_server.log",
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
    health = data.get("candidate_health") or {}
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
    candidate_log_tail = _tail(artifact_dir / "candidate_server.log", 80)

    decision = (
        "Bank P3-D as an opt-in server smoke only. Keep default promotion closed."
        if verdict == "PASS"
        else "Do not bank P3-D; keep the path opt-in and inspect the failed server gate."
    )

    lines = [
        "# Stage 6 Phase 3-D — OpenAI server RC smoke gate",
        "",
        f"Date: {report_date}",
        "",
        f"Verdict: **{verdict}** ({reason}).",
        "",
        "P3-D launches baseline and candidate OpenAI-compatible servers, then checks",
        "server surface, greedy text parity, non-degenerate responses, and candidate",
        "`/health` release/reload counters for the opt-in zero-shadow prefill path.",
        "",
        "**Boundary:** a PASS here is not default promotion and not full RC quality.",
        "Full MMLU/GPQA/tool/long-context publication gates remain separate.",
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
        f"| Preset | `{data.get('preset', 'unknown')}` |",
        f"| Model | `{data.get('model', 'unknown')}` |",
        f"| Max new tokens | `{data.get('max_new', 'unknown')}` |",
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
        f"| P3-C predecessor pass | `{_fmt_bool(passes.get('p3c_pass'))}` |",
        f"| Server surface | `{_fmt_bool(passes.get('server_surface'))}` |",
        f"| Prompt count | `{_fmt_bool(passes.get('prompt_count'))}` |",
        f"| Functional non-degenerate | `{_fmt_bool(passes.get('functional_non_degenerate'))}` |",
        f"| Server text exact | `{_fmt_bool(passes.get('server_text_exact'))}` |",
        f"| Release enabled | `{_fmt_bool(passes.get('release_enabled'))}` |",
        f"| Release consumed | `{_fmt_bool(passes.get('release_consumed'))}` |",
        f"| Decode shadows currently released | `{_fmt_bool(passes.get('decode_shadows_currently_released'))}` |",
        f"| Release meaningful | `{_fmt_bool(passes.get('release_meaningful'))}` |",
        f"| Reload observed | `{_fmt_bool(passes.get('reload_observed'))}` |",
        f"| Banked server smoke | `{data.get('banked_server_smoke')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        f"| Banked full RC quality | `{data.get('banked_full_rc_quality')}` |",
        f"| Release/reload count | `{health.get('release_reload_count')}` |",
        f"| Reload expected min | `{health.get('reload_expected_min')}` |",
        f"| Last release GiB | `{health.get('last_release_gib')}` |",
        f"| Last reload seconds | `{health.get('last_reload_seconds')}` |",
        "",
        "## Decision",
        "",
        decision,
        "",
        "Do not treat this report as full MMLU/GPQA/tool/long-context RC evidence.",
    ]
    if run_log_tail:
        lines.extend(["", "## Run Log Tail", "", "```text", run_log_tail, "```"])
    if candidate_log_tail:
        lines.extend(["", "## Candidate Server Log Tail", "", "```text", candidate_log_tail, "```"])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact_dir", help="Pulled P3-D artifact directory")
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
