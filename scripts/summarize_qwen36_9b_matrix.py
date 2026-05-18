#!/usr/bin/env python3
"""Summarize Qwen3.6 9B dense release matrix JSON into Markdown report.

Usage:
    python scripts/summarize_qwen36_9b_matrix.py \
        --json reports/qwen36_9b/qwen36_9b_dense_matrix_schema_v1.json \
        --out docs/QWEN36_9B_DENSE_RELEASE_MATRIX_20260518.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _fmt_float(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def _fmt_str(v: str | None) -> str:
    return v if v is not None else "—"


def _badge(e: dict[str, Any]) -> str:
    verdict = e.get("verdict", "")
    provisional = e.get("provisional", False)
    if provisional:
        return "🔶 PROVISIONAL"
    if verdict == "PENDING_SPARK":
        return "⏳ PENDING (Spark)"
    if verdict == "PENDING_R6000":
        return "⏳ PENDING (R6000)"
    if verdict == "REFERENCE":
        return "📌 REFERENCE"
    return verdict


def _render_dense_table(entries: list[dict[str, Any]]) -> str:
    lines = [
        "## 9B Dense Endpoint Matrix",
        "",
        "| Model | Quant | Runtime | Device | Size (GB) | MMLU | GPQA | Single TPS | Verdict |",
        "|-------|-------|---------|--------|-----------|------|------|------------|---------|",
    ]
    for e in entries:
        if e.get("arch") != "dense":
            continue
        lines.append(
            f"| {e['model_id']} | {e['quant']} | {e['runtime']} | {_fmt_str(e.get('device_class'))} "
            f"| {_fmt_float(e.get('size_gb'))} | {_fmt_float(e.get('mmlu'))} | {_fmt_float(e.get('gpqa'))} "
            f"| {_fmt_float(e.get('single_tps'))} | {_badge(e)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_reference_table(entries: list[dict[str, Any]]) -> str:
    lines = [
        "## 35B MoE Reference (high-quality serving candidate)",
        "",
        "| Model | Quant | Runtime | Device | Size (GB) | MMLU | GPQA | Single TPS | Verdict |",
        "|-------|-------|---------|--------|-----------|------|------|------------|---------|",
    ]
    for e in entries:
        if e.get("arch") != "moe":
            continue
        lines.append(
            f"| {e['model_id']} | {e['quant']} | {e['runtime']} | {_fmt_str(e.get('device_class'))} "
            f"| {_fmt_float(e.get('size_gb'))} | {_fmt_float(e.get('mmlu'))} | {_fmt_float(e.get('gpqa'))} "
            f"| {_fmt_float(e.get('single_tps'))} | {_badge(e)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_notes(entries: list[dict[str, Any]]) -> str:
    lines = ["## Context Notes", ""]
    for e in entries:
        if e.get("ctx_notes"):
            flag = "🔶" if e.get("provisional") else "•"
            lines.append(f"{flag} **{e['model_id']} ({e['quant']})**: {e['ctx_notes']}")
    lines.append("")
    return "\n".join(lines)


def generate_report(data: dict[str, Any]) -> str:
    entries = data.get("entries", [])
    created = data.get("created", "")
    note = data.get("note", "")

    lines = [
        "# Qwen3.6 9B Dense Release Matrix",
        "",
        f"**Date:** {created}",
        f"**Schema:** `{data.get('schema', 'unknown')}`",
        "",
        "> ⚠️ **Disclaimer:** This matrix is a living document. Numbers marked 🔶 PROVISIONAL are preliminary human benchmarks and must be confirmed by automated Spark/R6000 pipelines before release. Do not use provisional values as final marketing claims.",
        "",
        "## Product Positioning",
        "",
        "- **9B Dense** = endpoint / 16 GB VRAM / 端侧候选. 目标是在消费级 Mac 和 24 GB NVIDIA 卡上跑满 quality + speed.",
        "- **35B-A3B MoE** = 高质量服务候选. 目标是在 R6000 / B200 上提供 100+ TPS 的 NVFP4 服务.",
        "- **Q4_K_M / GGUF** = Mac + llama.cpp 阵营. 端侧首选, 质量可接受, 体积最小.",
        "- **NVFP4 / Lynn Engine** = NVIDIA / Blackwell 阵营. 服务端首选, 质量最高, TensorCore 加速.",
        "",
        _render_dense_table(entries),
        _render_reference_table(entries),
        _render_notes(entries),
        "## How to Update",
        "",
        "1. Edit `reports/qwen36_9b/qwen36_9b_dense_matrix_schema_v1.json` with new benchmark results.",
        "2. Run `bash scripts/qwen36_9b_dense_matrix.sh` to regenerate this Markdown.",
        "3. Backfill `mmlu` / `gpqa` / `single_tps` from Spark (Mac) or R6000 (NVIDIA) benchmark pipelines.",
        "4. Once a row is verified, set `provisional: false` and update `source` with pipeline reference.",
        "",
        "## Next Pipelines",
        "",
        "- **Spark (Mac/llama.cpp)**: Q4_K_M MMLU/GPQA full-suite; single-thread / concurrent TPS on M3 Max / M4 Max.",
        "- **R6000 (NVIDIA/Lynn)**: BF16 / NVFP4 MMLU/GPQA; decode TPS at 4K/32K context; concurrent batch.",
        "- **Atlas AGPL** (reference only, do not merge): Q4_K_M MMLU baseline for cross-check.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Qwen3.6 9B dense release matrix Markdown.")
    ap.add_argument("--json", required=True, help="Path to matrix schema JSON.")
    ap.add_argument("--out", required=True, help="Output Markdown path.")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text())
    md = generate_report(data)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)
    print(f"[matrix] Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
