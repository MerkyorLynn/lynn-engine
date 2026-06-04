#!/usr/bin/env python3
"""Write a Stage 6 P3-A report from a pulled artifact directory."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

from summarize_stage6_p3a_contract_probe import _verdict, summarize


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
    candidate = data.get("candidate") or {}
    shape = data.get("shape") or {}
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
    log_tail = _tail(artifact_dir / "run.log", 80)

    decision = (
        "Bank P3-A as a contract-shaped grouped active-MoE probe only. Do not promote P3 or claim a fused kernel."
        if verdict == "PASS"
        else "Do not bank P3-A; keep P3 in contract/probe status and investigate the failed gate."
    )

    lines = [
        "# Stage 6 Phase 3-A — grouped active-MoE contract probe",
        "",
        f"Date: {report_date}",
        "",
        f"Verdict: **{verdict}** ({reason}).",
        "",
        "P3-A tests the grouped active-MoE zero-shadow contract after P2-N/P2-O.",
        "It excludes router and shared expert from the measured candidate, builds a",
        "BF16 active-expert reference, deletes the active BF16 shadows, then runs",
        "`active_moe_grouped_prefill_p3a(...)` from packed NVFP4 tensors only.",
        "",
        "**Boundary:** this report cannot bank a fused P3 kernel. Even a PASS only",
        "banks the P3-A contract probe and its artifact schema.",
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
        f"| Layer | `{data.get('layer', 'unknown')}` |",
        f"| Batches | `{data.get('batches', 'unknown')}` |",
        f"| Candidate | `{candidate or 'default'}` |",
        f"| Shape | `H={shape.get('hidden')} I={shape.get('intermediate')} E={shape.get('num_experts')} top_k={shape.get('top_k')}` |",
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
        f"| Banked fused kernel flag is false | `{data.get('banked_fused_kernel') is False}` |",
        f"| Numeric | `{_fmt_bool(passes.get('numeric'))}` |",
        f"| Active BF16 shadow absent | `{_fmt_bool(passes.get('shadow_absent_at_candidate_start'))}` |",
        f"| Aggregate pass | `{_fmt_bool(passes.get('all'))}` |",
        f"| BF16 active expert bytes | `{bytes_.get('bf16_layer_active_experts', 'unknown')}` |",
        f"| Packed active expert bytes | `{bytes_.get('packed_layer_active_experts', 'unknown')}` |",
        f"| Inter scratch estimate | `{bytes_.get('max_inter_scratch_estimate', 'unknown')}` |",
        f"| Memory after deleting BF16 active GiB | `{bytes_.get('mem_after_deleting_bf16_active_gib', 'unknown')}` |",
        "",
        "## Decision",
        "",
        decision,
        "",
        "Do not treat P3-A as real-prompt, residual-stack, server, or RC quality proof.",
        "Those remain P3-B/P3-C/P3-D gates.",
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
    ap.add_argument("artifact_dir", help="Pulled P3-A artifact directory")
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
