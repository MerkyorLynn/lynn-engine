#!/usr/bin/env python3
"""Write a Stage 6 P3-C resident-prompt report from an artifact directory."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

from summarize_stage6_p3c_resident_prompt_gate import _verdict, summarize


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
    summary_md = artifact_dir / "summary.md"
    if summary_md.exists():
        summary = summary_md.read_text().strip()
    else:
        summary = summarize(data).strip()

    verdict, reason = _verdict(data)
    passes = data.get("passes") or {}
    memory = data.get("memory") or {}
    release = memory.get("release") or {}
    git_head = _read(artifact_dir / "git_head.txt", "unknown")
    expected_head = _read(artifact_dir / "expected_git_head.txt", "unknown")
    manifest = _read(artifact_dir / "provenance_manifest.txt", "missing")
    expected_manifest = _read(artifact_dir / "expected_provenance_manifest.txt", "missing")
    head_check = _read(artifact_dir / "head_check.txt", "missing")
    git_status = _read(artifact_dir / "git_status.txt", "")
    gpu_before = _read(artifact_dir / "nvidia_smi_before.txt", "missing")
    gpu_after = _read(artifact_dir / "nvidia_smi_after.txt", "missing")
    docker_exit = _read(artifact_dir / "docker_exit_code.txt", "missing")
    log_tail = _tail(artifact_dir / "run.log", 100)

    decision = (
        "Bank P3-C as a resident-runner real-prompt smoke only. Do not promote server/default/RC."
        if verdict == "PASS"
        else "Do not bank P3-C; keep the path opt-in and investigate the failed gate."
    )

    lines = [
        "# Stage 6 Phase 3-C — resident-runner real-prompt gate",
        "",
        f"Date: {report_date}",
        "",
        f"Verdict: **{verdict}** ({reason}).",
        "",
        "P3-C runs real prompts through the resident runner after active-MoE BF16",
        "expert shadows are released. Candidate prefill uses",
        "`LYNN_PACKED_PREFILL_SLOW_MODE=p3a_grouped` plus block linear-attn.",
        "",
        "**Boundary:** a PASS here is not a server/default/RC promotion. P3-D owns",
        "promotion and quality batteries.",
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
        f"| P3-B predecessor pass | `{_fmt_bool(passes.get('p3b_pass'))}` |",
        f"| Prompt count | `{_fmt_bool(passes.get('prompt_count'))}` |",
        f"| Functional non-degenerate | `{_fmt_bool(passes.get('functional_non_degenerate'))}` |",
        f"| Generated-token exact | `{_fmt_bool(passes.get('generated_token_exact', passes.get('token_exact')))}` |",
        f"| Release meaningful | `{_fmt_bool(passes.get('release_meaningful'))}` |",
        f"| Memory drop meaningful | `{_fmt_bool(passes.get('memory_drop_meaningful'))}` |",
        f"| Reload not called | `{_fmt_bool(passes.get('reload_not_called'))}` |",
        f"| Banked server path flag is false | `{data.get('banked_server_path') is False}` |",
        f"| Banked RC quality flag is false | `{data.get('banked_rc_quality') is False}` |",
        f"| Released tensors | `{release.get('released_tensors', 'unknown')}` |",
        f"| Released GiB | `{release.get('released_gib', 'unknown')}` |",
        f"| Memory drop GiB | `{memory.get('drop_gib', 'unknown')}` |",
        "",
        "## Decision",
        "",
        decision,
        "",
        "Do not treat this report as logits, hidden-state, full RC, or server default proof.",
    ]
    if log_tail:
        lines.extend([
            "",
            "## Run Log Tail",
            "",
            "```text",
            log_tail,
            "```",
        ])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact_dir", help="Pulled P3-C artifact directory")
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
