#!/usr/bin/env python3
"""
summarize_qwen35_9b_r6000_reports.py
Render Markdown from the unified Qwen3.5-9B R6000 benchmark summary JSON.

Usage:
    python3 summarize_qwen35_9b_r6000_reports.py \
        --summary reports/qwen35_9b/r6000_qwen35_9b_official_matrix_summary_*.json \
        --out docs/QWEN35_9B_R6000_NVFP4_PIPELINE_20260518.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Failed to load {path}: {exc}", file=sys.stderr)
        return None


def fmt_val(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✅ Yes" if v else "❌ No"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def fmt_metric(metric: dict) -> str:
    status = metric.get("status", "PENDING")
    score = metric.get("score")
    correct = metric.get("correct")
    total = metric.get("total")
    if status == "DONE" and score is not None:
        parts = [f"score={fmt_val(score)}"]
        if correct is not None:
            parts.append(f"correct={fmt_val(correct)}")
        if total is not None:
            parts.append(f"total={fmt_val(total)}")
        return "✅ " + " / ".join(parts)
    if status == "BLOCKED":
        br = metric.get("blocked_reason", "")
        return f"🔴 BLOCKED: {br}" if br else "🔴 BLOCKED"
    return "⏳ PENDING"


def fmt_tps(tps: dict) -> str:
    status = tps.get("status", "PENDING")
    if status == "DONE":
        parts = []
        for k in ("tps_128", "tps_256", "tps_512"):
            v = tps.get(k)
            if v is not None:
                parts.append(f"{k.replace('tps_', '')}={fmt_val(v)}")
        return "✅ " + " / ".join(parts) if parts else "✅ DONE (no data)"
    if status == "BLOCKED":
        br = tps.get("blocked_reason", "")
        return f"🔴 BLOCKED: {br}" if br else "🔴 BLOCKED"
    return "⏳ PENDING"


def render_markdown(summary: dict) -> str:
    lines: list[str] = []
    model_id = summary.get("model_id", "Qwen3.5-9B")
    stamp = summary.get("stamp", "unknown")
    schema = summary.get("schema", "unknown")

    lines.append(f"# {model_id} R6000 NVFP4 Pipeline Report")
    lines.append("")
    lines.append(f"**Generated:** {stamp}  ")
    lines.append(f"**Schema:** {schema}  ")
    lines.append("")
    lines.append("> **Note:** Official Qwen3.5-9B route (not 3.6-9B-Dense). ")
    lines.append("> Covers BF16 baseline, Lynn-native W4A16 NVFP4, and Q4_K_M (PENDING if missing).")
    lines.append("")

    # Asset table
    assets = summary.get("assets", {})
    lines.append("## Asset Status")
    lines.append("")
    lines.append("| Asset | Ready | Size (GiB) | Details |")
    lines.append("|-------|-------|------------|---------|")

    bf16 = assets.get("bf16", {})
    lines.append(
        f"| BF16 | {fmt_val(bf16.get('ready'))} | {fmt_val(bf16.get('gib'))} | bytes={fmt_val(bf16.get('bytes'))} |"
    )

    nvfp4 = assets.get("nvfp4", {})
    nvfp4_details = []
    if nvfp4.get("quantized_count") is not None:
        nvfp4_details.append(f"quantized={nvfp4['quantized_count']}")
    if nvfp4.get("kept_count") is not None:
        nvfp4_details.append(f"kept={nvfp4['kept_count']}")
    if nvfp4.get("output_shards") is not None:
        nvfp4_details.append(f"shards={nvfp4['output_shards']}")
    if nvfp4.get("pack_elapsed_seconds") is not None:
        nvfp4_details.append(f"elapsed={nvfp4['pack_elapsed_seconds']}s")
    lines.append(
        f"| NVFP4 | {fmt_val(nvfp4.get('ready'))} | {fmt_val(nvfp4.get('gib'))} | "
        f"{' '.join(nvfp4_details) or '—'} |"
    )

    q4km = assets.get("q4_k_m", {})
    lines.append(
        f"| Q4_K_M | {fmt_val(q4km.get('ready'))} | — | "
        f"{q4km.get('status', 'PENDING')}: {q4km.get('note', '')} |"
    )
    lines.append("")

    # Quality table
    results = summary.get("results", {})
    lines.append("## Quality Metrics")
    lines.append("")
    lines.append("| Quant | Status | MMLU-500-5shot | GPQA-diamond |")
    lines.append("|-------|--------|----------------|--------------|")
    for quant in ("bf16", "nvfp4", "q4_k_m"):
        r = results.get(quant, {})
        lines.append(
            f"| {quant.upper()} | {r.get('status', 'PENDING')} | "
            f"{fmt_metric(r.get('mmlu_500_5shot', {}))} | "
            f"{fmt_metric(r.get('gpqa_diamond', {}))} |"
        )
    lines.append("")

    # Performance table
    lines.append("## Performance (Single TPS)")
    lines.append("")
    lines.append("| Quant | Status | TPS-128 | TPS-256 | TPS-512 | Load (s) | Size (GiB) |")
    lines.append("|-------|--------|---------|---------|---------|----------|------------|")
    for quant in ("bf16", "nvfp4", "q4_k_m"):
        r = results.get(quant, {})
        tps = r.get("single_tps", {})
        lines.append(
            f"| {quant.upper()} | {r.get('status', 'PENDING')} | "
            f"{fmt_val(tps.get('tps_128'))} | {fmt_val(tps.get('tps_256'))} | {fmt_val(tps.get('tps_512'))} | "
            f"{fmt_val(r.get('load_seconds'))} | {fmt_val(r.get('size_gib'))} |"
        )
    lines.append("")

    # Detailed per-metric status
    lines.append("## Metric Details")
    lines.append("")
    for quant in ("bf16", "nvfp4", "q4_k_m"):
        r = results.get(quant, {})
        lines.append(f"### {quant.upper()}")
        lines.append("")
        lines.append(f"- **Overall status:** {r.get('status', 'PENDING')}")
        if r.get("blocked_reason"):
            lines.append(f"- **Blocked reason:** {r['blocked_reason']}")

        mmlu = r.get("mmlu_500_5shot", {})
        lines.append(f"- **MMLU-500-5shot:** {fmt_metric(mmlu)}")
        if mmlu.get("report_path"):
            lines.append(f"  - Report: `{mmlu['report_path']}`")

        gpqa = r.get("gpqa_diamond", {})
        lines.append(f"- **GPQA-diamond:** {fmt_metric(gpqa)}")
        if gpqa.get("report_path"):
            lines.append(f"  - Report: `{gpqa['report_path']}`")

        tps = r.get("single_tps", {})
        lines.append(f"- **Single TPS:** {fmt_tps(tps)}")
        if tps.get("report_path"):
            lines.append(f"  - Report: `{tps['report_path']}`")

        lines.append(f"- **Size:** {fmt_val(r.get('size_gib'))} GiB")
        lines.append(f"- **Load+eval time:** {fmt_val(r.get('load_seconds'))} s")
        lines.append("")

    # Product positioning
    lines.append("## Product Positioning")
    lines.append("")
    lines.append("- **Qwen3.5-9B** = official endpoint / 16G VRAM branch candidate.")
    lines.append("- **BF16** = quality ceiling; ~18 GiB. Reference for all quantized variants.")
    lines.append("- **NVFP4 (W4A16)** = NVIDIA Blackwell serving path; ~5.5 GiB.")
    lines.append("  Lynn-native packed decode + TensorCore MMA on SM100+.")
    lines.append("- **Q4_K_M** = Mac / Apple Silicon / llama.cpp path; ~5.2 GiB.")
    lines.append("  Left as PENDING in this pipeline; download separately if needed.")
    lines.append("")

    lines.append("## Pipeline Stages")
    lines.append("")
    lines.append("| Stage | Description | Status |")
    lines.append("|-------|-------------|--------|")
    lines.append("| Asset check | BF16 index + NVFP4 manifest + Q4_K_M GGUF | — |")
    lines.append("| Size summary | GiB + manifest fields | — |")
    lines.append("| GPU idle check | nvidia-smi / SKIP_GPU | — |")
    lines.append("| MMLU-500-5shot | Per-quant MMLU eval | See table |")
    lines.append("| GPQA-diamond | Per-quant GPQA eval | See table |")
    lines.append("| Single TPS | P25 probe 128/256/512 | See table |")
    lines.append("| Summarize | Markdown report | — |")
    lines.append("")

    lines.append("---")
    lines.append("*Report generated by `summarize_qwen35_9b_r6000_reports.py`*")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Qwen3.5-9B R6000 summary Markdown")
    parser.add_argument("--summary", required=True, help="Path to summary JSON")
    parser.add_argument("--out", required=True, help="Output Markdown path")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    out_path = Path(args.out)

    summary = load_json(summary_path)
    if summary is None:
        summary = {
            "schema": "lynn-qwen35-9b-official-matrix-summary-v1",
            "model_id": "Qwen3.5-9B",
            "arch": "dense",
            "stamp": "unknown",
            "assets": {
                "bf16": {"ready": False, "gib": None},
                "nvfp4": {"ready": False, "gib": None},
                "q4_k_m": {"ready": False, "status": "PENDING"},
            },
            "results": {
                "bf16": {"status": "NO_DATA", "blocked_reason": "Summary JSON not found"},
                "nvfp4": {"status": "NO_DATA", "blocked_reason": "Summary JSON not found"},
                "q4_k_m": {"status": "NO_DATA", "blocked_reason": "Summary JSON not found"},
            },
        }
        print(f"[WARN] Summary not found; rendering scaffold: {summary_path}", file=sys.stderr)

    md = render_markdown(summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[matrix] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
