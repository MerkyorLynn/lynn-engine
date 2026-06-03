#!/usr/bin/env python3
"""Write a Stage 6 P3-B selected-prefill report from an artifact directory."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

from summarize_stage6_p3b_selected_prefill_gate import _verdict, summarize


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
    bytes_ = data.get("bytes") or {}
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
        "Bank P3-B selected-prefill composition only. Do not claim fused-kernel, server, or RC promotion."
        if verdict == "PASS"
        else "Do not bank P3-B; keep P3-B in failed/pending gate status and investigate the failed evidence row."
    )

    lines = [
        "# Stage 6 Phase 3-B — selected-prefill composition gate",
        "",
        f"Date: {report_date}",
        "",
        f"Verdict: **{verdict}** ({reason}).",
        "",
        "P3-B places the P3-A grouped active-MoE contract inside a selected",
        "transformer prefill stack. It compares BF16 prefill, P2-N reference",
        "(`p2e_hybrid` + block linear-attn), and P3-B candidate",
        "(`p3a_grouped` + block linear-attn).",
        "",
        "**Boundary:** a PASS here only banks selected-layer composition. It does",
        "not bank a fused grouped-MoE kernel, server path, RC quality, or default",
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
        f"| Model | `{data.get('model', 'unknown')}` |",
        f"| Layers | `{data.get('layers', 'unknown')}` |",
        f"| Layer types | `{data.get('layer_types', 'unknown')}` |",
        f"| Seq lens | `{data.get('seq_lens', 'unknown')}` |",
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
        f"| Predecessors pass | `{_fmt_bool(passes.get('predecessors_pass'))}` |",
        f"| Numeric | `{_fmt_bool(passes.get('numeric'))}` |",
        f"| Final stack cosine min | `{passes.get('final_stack_cosine_min', 'unknown')}` |",
        f"| Final stack argmax | `{_fmt_bool(passes.get('final_stack_argmax_match'))}` |",
        f"| Active BF16 shadow absent | `{_fmt_bool(passes.get('no_active_bf16_shadow'))}` |",
        f"| Reload not called | `{_fmt_bool(passes.get('reload_not_called'))}` |",
        f"| Speed vs P2-N reference | `{_fmt_bool(passes.get('speed_vs_p2n_reference'))}` |",
        f"| Banked fused kernel flag is false | `{data.get('banked_fused_kernel') is False}` |",
        f"| Banked server path flag is false | `{data.get('banked_server_path') is False}` |",
        f"| BF16 active expert bytes | `{bytes_.get('bf16_active_experts', 'unknown')}` |",
        f"| Packed active expert bytes | `{bytes_.get('packed_active_experts', 'unknown')}` |",
        f"| Memory after deleting active BF16 GiB | `{bytes_.get('mem_after_deleting_bf16_active_gib', 'unknown')}` |",
        "",
        "## Decision",
        "",
        decision,
        "",
        "P3-C server integration and P3-D/RC quality remain separate gates.",
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
    ap.add_argument("artifact_dir", help="Pulled P3-B artifact directory")
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
