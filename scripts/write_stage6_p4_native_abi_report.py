#!/usr/bin/env python3
"""Write a Stage 6 P4 native fused-MoE ABI preflight report."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

from summarize_stage6_p4_native_abi_preflight import _verdict, summarize


def _read(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
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
    data = json.loads(result_path.read_text(encoding="utf-8"))
    summary_path = artifact_dir / "summary.md"
    summary = summary_path.read_text(encoding="utf-8").strip() if summary_path.exists() else summarize(data).strip()
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
    decision = (
        "Bank P4 native ABI preflight only. Keep fused kernel and default promotion closed."
        if verdict == "PASS"
        else "Do not bank P4 ABI preflight; inspect the failed build/symbol/guard gate."
    )

    lines = [
        "# Stage 6 Phase 4 - native fused-MoE ABI preflight",
        "",
        f"Date: {report_date}",
        "",
        f"Verdict: **{verdict}** ({reason}).",
        "",
        "P4 verifies the C++/CUDA hot-path boundary for the future fused 4-bit",
        "zero-shadow active-MoE kernel. It does not bank a fused kernel or default",
        "promotion.",
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
        f"| Symbol | `{data.get('symbol', 'unknown')}` |",
        f"| Decision | `{data.get('decision', 'unknown')}` |",
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
        f"| Extension loaded | `{_fmt_bool(passes.get('extension_loaded'))}` |",
        f"| Symbol present | `{_fmt_bool(passes.get('symbol_present'))}` |",
        f"| Reference output returned | `{_fmt_bool(passes.get('reference_output_returned'))}` |",
        f"| Output finite | `{_fmt_bool(passes.get('output_finite'))}` |",
        f"| Zero-shadow ABI | `{_fmt_bool(passes.get('zero_shadow_abi'))}` |",
        f"| Packed byte budget | `{_fmt_bool(passes.get('packed_byte_budget'))}` |",
        f"| Aggregate | `{_fmt_bool(passes.get('all'))}` |",
        f"| Banked native ABI preflight | `{data.get('banked_native_abi_preflight')}` |",
        f"| Banked fused kernel | `{data.get('banked_fused_kernel')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        "",
        "## Decision",
        "",
        decision,
    ]
    if run_log_tail:
        lines.extend(["", "## Run Log Tail", "", "```text", run_log_tail, "```"])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact_dir", help="Pulled P4 artifact directory")
    ap.add_argument("--report-out", required=True, help="Markdown report output path")
    ap.add_argument("--date", default=str(_dt.date.today()))
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    report = write_report(artifact_dir, report_date=args.date)
    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
