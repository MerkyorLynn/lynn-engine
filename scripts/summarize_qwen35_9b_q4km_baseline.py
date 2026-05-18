#!/usr/bin/env python3
"""
summarize_qwen35_9b_q4km_baseline.py
Render Markdown from the Qwen3.5-9B Q4_K_M llama.cpp baseline JSON report.

Aligned with Lynn 9B NVFP4 watcher fields for cross-platform comparison.

Usage:
    python3 summarize_qwen35_9b_q4km_baseline.py \
        --report reports/qwen35_9b/r6000_qwen35_9b_q4km_baseline_*.json \
        --output docs/QWEN35_9B_Q4KM_LLAMA_BASELINE_20260519.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Failed to load {path}: {exc}", file=sys.stderr)
        return None


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✅" if v else "❌"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _tps_cell(entry: dict | None) -> str:
    if entry is None:
        return "⏳ PENDING"
    if not entry.get("ok", False):
        err = entry.get("error", "unknown")
        return f"🔴 FAIL: {err}"
    tps = entry.get("wall_tps") or entry.get("batch_wall_tps", 0)
    return f"**{tps:.1f}** TPS"


def _lc_cell(entry: dict | None) -> str:
    if entry is None:
        return "⏳ PENDING"
    if not entry.get("ok", False):
        err = entry.get("error", "")
        if err and ("oom" in err.lower() or "out of memory" in err.lower()):
            return "🔴 OOM"
        return f"🔴 FAIL: {err}"
    tps = entry.get("wall_tps", 0)
    pt = entry.get("prompt_tokens", 0)
    return f"**{tps:.1f}** TPS ({pt} tok)"


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    status = report.get("status", "UNKNOWN")
    model_id = report.get("model_id", "Qwen3.5-9B")
    quant = report.get("quant", "Q4_K_M")
    model_path = report.get("model_path")
    size_gib = report.get("size_gib")
    llama_bin = report.get("llama_cpp_binary")
    git_rev = report.get("git_rev")
    errors = report.get("errors", [])

    # Status icon
    if status == "PENDING_DOWNLOAD":
        icon = "⏳"
    elif status == "DONE":
        icon = "🟢"
    else:
        icon = "🔴"

    lines.append(f"# {model_id} {quant} llama.cpp Baseline — {icon} {status}")
    lines.append("")
    lines.append(f"**Model:** `{model_id}` | **Quant:** `{quant}`")
    if model_path:
        lines.append(f"**Path:** `{model_path}`")
    if size_gib:
        lines.append(f"**Size:** {size_gib} GiB")
    if llama_bin:
        lines.append(f"**Binary:** `{llama_bin}`")
    if git_rev:
        lines.append(f"**llama.cpp rev:** `{git_rev}`")
    lines.append("")

    # --- Errors ---
    if errors:
        lines.append("## Errors")
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")

    # --- PENDING_DOWNLOAD recommendation ---
    if status == "PENDING_DOWNLOAD":
        rec = report.get("recommendation", {})
        cmds = rec.get("download_commands", [])
        if cmds:
            lines.append("## Recommended Download")
            lines.append("")
            lines.append("```bash")
            lines.extend(cmds)
            lines.append("```")
            lines.append("")
        lines.append("## Pending Benchmarks")
        lines.append("")
        lines.append("| Section | Status |")
        lines.append("|---------|--------|")
        lines.append("| Single TPS (128/256/512) | ⏳ PENDING |")
        lines.append("| Concurrent TPS (2/4/8) | ⏳ PENDING |")
        lines.append("| Long Context (4k/16k/32k) | ⏳ PENDING |")
        lines.append("")
        return "\n".join(lines)

    # --- Single TPS Table ---
    single = report.get("single_tps", {})
    lines.append("## Single-Stream Decode TPS")
    lines.append("")
    lines.append("| max_tokens | TPS | prompt_tokens | completion_tokens | elapsed_s |")
    lines.append("|------------|-----|---------------|-------------------|-----------|")
    for mt in sorted(single.keys(), key=lambda x: int(x)):
        e = single[mt]
        if e.get("ok"):
            lines.append(
                f"| {mt} | **{e.get('wall_tps', 0):.1f}** | "
                f"{e.get('prompt_tokens', 0)} | "
                f"{e.get('completion_tokens', 0)} | "
                f"{e.get('elapsed_s', 0):.3f} |"
            )
        else:
            lines.append(f"| {mt} | 🔴 FAIL | — | — | — |")
    lines.append("")

    # --- Concurrent TPS Table ---
    conc = report.get("concurrent_tps", {})
    lines.append("## Concurrent Total TPS")
    lines.append("")
    lines.append("| concurrency | batch_wall_tps | elapsed_s | errors |")
    lines.append("|-------------|----------------|-----------|--------|")
    for c in sorted(conc.keys(), key=lambda x: int(x)):
        e = conc[c]
        ok = e.get("ok", False)
        tps = e.get("batch_wall_tps", 0)
        elapsed = e.get("elapsed_s", 0)
        errs = e.get("errors", [])
        err_str = "; ".join(errs[:2]) if errs else "none"
        if ok:
            lines.append(f"| {c} | **{tps:.1f}** | {elapsed:.3f} | {err_str} |")
        else:
            lines.append(f"| {c} | 🔴 FAIL | — | {err_str} |")
    lines.append("")

    # --- Long Context Table ---
    lc = report.get("long_context", {})
    lines.append("## Long-Context Smoke (prefill + decode)")
    lines.append("")
    lines.append("| chars | prompt_tokens | wall_tps | completion_tokens | elapsed_s | status |")
    lines.append("|-------|---------------|----------|-------------------|-----------|--------|")
    for chars in sorted(lc.keys(), key=lambda x: int(x)):
        e = lc[chars]
        ok = e.get("ok", False)
        tps = e.get("wall_tps", 0)
        pt = e.get("prompt_tokens", 0)
        ct = e.get("completion_tokens", 0)
        elapsed = e.get("elapsed_s", 0)
        err = e.get("error")
        if ok:
            lines.append(
                f"| {chars} | {pt} | **{tps:.1f}** | {ct} | {elapsed:.3f} | ✅ |"
            )
        else:
            err_short = (err[:40] + "…") if err and len(err) > 40 else (err or "fail")
            is_oom = err and "oom" in err.lower() if err else False
            icon = "🔴 OOM" if is_oom else f"🔴 {err_short}"
            lines.append(f"| {chars} | {pt} | — | — | — | {icon} |")
    lines.append("")

    # --- Lynn NVFP4 cross-reference ---
    lines.append("## Cross-Reference: Lynn 9B NVFP4 Watcher Fields")
    lines.append("")
    lines.append(
        "The following maps Q4_K_M llama.cpp results to Lynn NVFP4 watcher fields "
        "for direct comparison on the same R6000 hardware."
    )
    lines.append("")
    lines.append("| Metric | Lynn NVFP4 Field | Q4_K_M Value |")
    lines.append("|--------|-------------------|--------------|")

    # single_tps_512
    s512 = single.get("512")
    if s512 and s512.get("ok"):
        lines.append(f"| Single TPS (512 tok) | `single_tps` | **{s512['wall_tps']:.1f}** TPS |")
    else:
        lines.append("| Single TPS (512 tok) | `single_tps` | ⏳ PENDING |")

    # concurrent_tps_8
    c8 = conc.get("8")
    if c8 and c8.get("ok"):
        lines.append(f"| Concurrent TPS (×8) | `concurrent_total_tps` | **{c8['batch_wall_tps']:.1f}** TPS |")
    else:
        lines.append("| Concurrent TPS (×8) | `concurrent_total_tps` | ⏳ PENDING |")

    # long_context_32k
    lc32k = lc.get("32768")
    if lc32k and lc32k.get("ok"):
        lines.append(f"| Long Context (32k) | `long_context_ok` | ✅ {lc32k['wall_tps']:.1f} TPS |")
    elif lc32k:
        err = lc32k.get("error", "")
        if err and "oom" in err.lower():
            lines.append("| Long Context (32k) | `long_context_ok` | 🔴 OOM |")
        else:
            lines.append("| Long Context (32k) | `long_context_ok` | 🔴 FAIL |")
    else:
        lines.append("| Long Context (32k) | `long_context_ok` | ⏳ PENDING |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Qwen3.5-9B Q4_K_M baseline")
    parser.add_argument("--report", required=True, help="Path to baseline JSON report")
    parser.add_argument("--output", default="", help="Output Markdown path")
    args = parser.parse_args()

    report_path = Path(args.report)
    report = _load_json(report_path)
    if report is None:
        print(f"[ERR] Cannot load report: {report_path}", file=sys.stderr)
        sys.exit(1)

    md = render_markdown(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"[OK] Written: {out_path}")
    else:
        print(md)


if __name__ == "__main__":
    main()
