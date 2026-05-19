#!/usr/bin/env python3
"""summarize_qwen35_9b_q4km_cuda_baseline.py

Converts R6000 Qwen3.5-9B Q4_K_M-imatrix CUDA baseline JSON reports into a
Markdown summary with cross-reference to Lynn NVFP4 watcher fields.

Usage:
    python3 scripts/summarize_qwen35_9b_q4km_cuda_baseline.py reports/qwen35_9b/*.json
    python3 scripts/summarize_qwen35_9b_q4km_cuda_baseline.py reports/qwen35_9b/ --latest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ── Lynn NVFP4 watcher reference (R6000) ─────────────────────────────────────
WATCHER_REF = {
    "single_tps_512": {"field": "single_tps (P25 probe)", "typical": "~104-107 TPS"},
    "concurrent_8_total": {"field": "concurrent_total_tps", "typical": "~380-400 TPS"},
    "gpqa": {"field": "gpqa_score", "typical": "~50.0%"},
}


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt_tps(v: float) -> str:
    if v == 0:
        return "—"
    return f"{v:,.1f}"


def fmt_pct(v: float) -> str:
    if v == 0:
        return "—"
    return f"{v:.1f}%"


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    status = report.get("status", "UNKNOWN")
    model_id = report.get("model_id", "?")
    quant = report.get("quant", "?")
    engine = report.get("engine", "llama.cpp CUDA")
    engine_detail = report.get("engine_detail", "")
    model_path = report.get("model_path", "?")
    size_gib = report.get("size_gib", "?")
    binary = report.get("llama_cpp_binary", "?")
    git_rev = report.get("git_rev", "?")
    n_gpu = report.get("n_gpu_layers", "?")
    ctx_size = report.get("ctx_size", "?")
    timestamp = report.get("timestamp", "?")

    lines.append(f"# R6000 {model_id} {quant} CUDA Baseline")
    lines.append("")
    lines.append(f"**Date:** {timestamp[:8] if len(timestamp) >= 8 else timestamp}")
    lines.append(f"**Status:** {'🟢' if status == 'DONE' else '🔴'} {status}")
    lines.append(f"**Engine:** {engine} ({engine_detail})")
    lines.append(f"**Model:** `{model_path}` ({size_gib} GiB)")
    lines.append(f"**Binary:** `{binary}` (rev `{git_rev}`)")
    lines.append(f"**GPU layers:** {n_gpu} | **Context:** {ctx_size}")
    lines.append("")

    if status == "PENDING_DOWNLOAD":
        lines.append("## GGUF Download Required")
        lines.append("")
        cmds = report.get("download_commands", {})
        for name, cmd in cmds.items():
            lines.append(f"**{name}:**")
            lines.append(f"```bash\n{cmd}\n```")
            lines.append("")
        errors = report.get("errors", [])
        if errors:
            lines.append(f"Errors: {', '.join(errors)}")
        return "\n".join(lines)

    # ── Single-stream TPS ──────────────────────────────────────────────────
    single = report.get("single_tps", {})
    if single:
        lines.append("## Single-Stream Decode TPS")
        lines.append("")
        lines.append("| max_tokens | Prompt tokens | Completion tokens | Wall TPS | Elapsed (s) |")
        lines.append("|---:|---:|---:|---:|---:|")
        for mt in sorted(single.keys(), key=int):
            d = single[mt]
            if d.get("ok"):
                lines.append(
                    f"| {mt} | {d.get('prompt_tokens', 0)} | "
                    f"{d.get('completion_tokens', 0)} | "
                    f"**{fmt_tps(d.get('wall_tps', 0))}** | "
                    f"{d.get('elapsed_s', 0):.3f} |"
                )
            else:
                lines.append(f"| {mt} | — | — | ❌ {d.get('error', 'failed')} | — |")
        lines.append("")

    # ── Concurrent TPS ─────────────────────────────────────────────────────
    concurrent = report.get("concurrent_tps", {})
    if concurrent:
        lines.append("## Concurrent Decode TPS (total)")
        lines.append("")
        lines.append("| Concurrency | Total completion tokens | Batch wall TPS | Elapsed (s) |")
        lines.append("|---:|---:|---:|---:|")
        for cc in sorted(concurrent.keys(), key=int):
            d = concurrent[cc]
            if d.get("ok"):
                lines.append(
                    f"| {cc} | {d.get('total_completion_tokens', 0)} | "
                    f"**{fmt_tps(d.get('batch_wall_tps', 0))}** | "
                    f"{d.get('elapsed_s', 0):.3f} |"
                )
            else:
                lines.append(f"| {cc} | — | ❌ {d.get('error', 'failed')} | — |")
        lines.append("")

    # ── Long-context ───────────────────────────────────────────────────────
    long_ctx = report.get("long_context", {})
    if long_ctx:
        lines.append("## Long-Context Prefill + Decode")
        lines.append("")
        lines.append("| Chars | Prompt tokens | Completion tokens | Wall TPS | Elapsed (s) | Status |")
        lines.append("|---:|---:|---:|---:|---:|:---:|")
        for lc in sorted(long_ctx.keys(), key=int):
            d = long_ctx[lc]
            ok = d.get("ok", False)
            status_icon = "✅" if ok else "❌"
            if ok:
                lines.append(
                    f"| {lc:,} | {d.get('total_prompt_tokens', 0):,} | "
                    f"{d.get('total_completion_tokens', 0)} | "
                    f"**{fmt_tps(d.get('wall_tps', 0))}** | "
                    f"{d.get('elapsed_s', 0):.3f} | {status_icon} |"
                )
            else:
                lines.append(
                    f"| {lc:,} | — | — | — | — | "
                    f"{status_icon} {d.get('error', 'failed')} |"
                )
        lines.append("")

    # ── GPQA ───────────────────────────────────────────────────────────────
    gpqa = report.get("gpqa", {})
    if gpqa:
        lines.append("## GPQA Diamond (32K Thinking)")
        lines.append("")
        lines.append(f"- **Accuracy:** {fmt_pct(gpqa.get('accuracy', 0))} ({gpqa.get('correct', 0)}/{gpqa.get('total', 0)})")
        lines.append(f"- **Engine:** {gpqa.get('engine', 'CUDA')}")
        lines.append(f"- **Thinking context:** {gpqa.get('thinking_chars', '?')} chars")
        lines.append("")

    # ── Cross-reference ────────────────────────────────────────────────────
    lines.append("## Cross-Reference: Lynn NVFP4 Watcher (R6000)")
    lines.append("")
    lines.append("| Metric | Lynn NVFP4 Watcher | Typical | This CUDA Baseline | Delta |")
    lines.append("|--------|-------------------|---------|-------------------|-------|")

    # Single 512 TPS
    s512 = single.get("512", {})
    s512_tps = s512.get("wall_tps", 0) if s512.get("ok") else 0
    typical_512 = 105.5  # midpoint of ~104-107
    delta_512 = s512_tps - typical_512 if s512_tps else 0
    lines.append(
        f"| Single 512 TPS | {WATCHER_REF['single_tps_512']['field']} | "
        f"{WATCHER_REF['single_tps_512']['typical']} | "
        f"**{fmt_tps(s512_tps)}** | "
        f"{'+' if delta_512 >= 0 else ''}{delta_512:.1f} |"
    )

    # Concurrent 8
    c8 = concurrent.get("8", {})
    c8_tps = c8.get("batch_wall_tps", 0) if c8.get("ok") else 0
    typical_c8 = 390
    delta_c8 = c8_tps - typical_c8 if c8_tps else 0
    lines.append(
        f"| Concurrent 8 total | {WATCHER_REF['concurrent_8_total']['field']} | "
        f"{WATCHER_REF['concurrent_8_total']['typical']} | "
        f"**{fmt_tps(c8_tps)}** | "
        f"{'+' if delta_c8 >= 0 else ''}{delta_c8:.1f} |"
    )

    # GPQA
    gpqa_acc = gpqa.get("accuracy", 0) if gpqa else 0
    typical_gpqa = 50.0
    delta_gpqa = gpqa_acc - typical_gpqa if gpqa_acc else 0
    lines.append(
        f"| GPQA accuracy | {WATCHER_REF['gpqa']['field']} | "
        f"{WATCHER_REF['gpqa']['typical']} | "
        f"**{fmt_pct(gpqa_acc)}** | "
        f"{'+' if delta_gpqa >= 0 else ''}{delta_gpqa:.1f}pp |"
    )
    lines.append("")

    # ── Errors ─────────────────────────────────────────────────────────────
    errors = report.get("errors", [])
    if errors:
        lines.append("## Errors")
        lines.append("")
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")

    return "\n".join(lines)


def find_latest_json(directory: str) -> str | None:
    """Find the most recent CUDA baseline JSON in a directory."""
    candidates = []
    for f in Path(directory).glob("*cuda*baseline*.json"):
        if f.is_file():
            candidates.append(f)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="JSON report files")
    parser.add_argument("--latest", action="store_true",
                        help="Process latest CUDA baseline in reports/qwen35_9b/")
    parser.add_argument("--dir", default="reports/qwen35_9b",
                        help="Report directory (with --latest)")
    args = parser.parse_args()

    paths = list(args.paths)
    if args.latest:
        latest = find_latest_json(args.dir)
        if latest:
            paths = [latest]
        else:
            print(f"No CUDA baseline reports found in {args.dir}", file=sys.stderr)
            sys.exit(1)

    if not paths:
        print("No report files specified. Use --latest or provide paths.", file=sys.stderr)
        sys.exit(1)

    for path in paths:
        try:
            report = load_report(path)
            md = render_markdown(report)
            print(md)
            if len(paths) > 1:
                print("\n---\n")
        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
