#!/usr/bin/env python3
"""P191: Summarize Qwen3.5-9B true-FP8 layer-mask sweep results.

Reads p190 JSON reports and classifies each candidate layer mask:
  EXACT_FAST:        all exact AND speedup >= 1.03
  EXACT_FLAT:        all exact AND speedup < 1.03
  AMBER_LATE_DRIFT:  not exact but all drift indices >= 32
  CLOSED_EARLY_DRIFT: any drift index < 32

Usage:
  python scripts/summarize_qwen35_9b_true_fp8_layer_masks.py \
      --reports /path/to/p190_*.json \
      --out-json summary.json \
      --out-md summary.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


def _extract_layer_mask(report: dict, filename: str) -> str:
    """Extract layer mask from report env or filename."""
    # Try report env
    env = report.get("candidate_env", report.get("env", {}))
    if isinstance(env, dict):
        mask = env.get("LYNN_TRUE_FP8_LAYER_MASK", "")
        if mask:
            return mask
    # Try filename pattern like ..._layers_0_3_7_...
    m = re.search(r"layers?[_-]([\d_]+)", filename)
    if m:
        return m.group(1).replace("_", ",")
    # Try from report top-level
    mask = report.get("layer_mask", report.get("candidate_label", ""))
    if mask:
        return str(mask)
    return "unknown"


def _classify(exact_count: int, total: int, drift_indices: list[int], speedup: float) -> str:
    """Classify a candidate based on exactness and speed."""
    all_exact = exact_count == total and total > 0
    if all_exact:
        return "EXACT_FAST" if speedup >= 1.03 else "EXACT_FLAT"
    if not drift_indices:
        return "CLOSED_EARLY_DRIFT"
    if all(d >= 32 for d in drift_indices):
        return "AMBER_LATE_DRIFT"
    return "CLOSED_EARLY_DRIFT"


def summarize_report(report: dict, filename: str) -> dict[str, Any]:
    """Extract summary from a single p190 report."""
    comparison = report.get("comparison", {})
    exact_count = comparison.get("exact_count", 0)
    total = comparison.get("total", 0)

    # Extract drift indices
    drift_indices = []
    rows = comparison.get("rows", report.get("results", []))
    for row in rows:
        if not row.get("exact", True):
            idx = row.get("first_drift_index", row.get("first_drift", None))
            if idx is not None:
                drift_indices.append(int(idx))

    # TPS
    ref_summary = report.get("reference_summary", {})
    cand_summary = report.get("candidate_summary", {})
    ref_tps = ref_summary.get("decode_tps_mean", 0.0)
    cand_tps = cand_summary.get("decode_tps_mean", 0.0)
    speedup = cand_tps / ref_tps if ref_tps > 0 else 0.0

    layer_mask = _extract_layer_mask(report, filename)
    verdict = _classify(exact_count, total, drift_indices, speedup)

    return {
        "filename": os.path.basename(filename),
        "layer_mask": layer_mask,
        "exact_count": exact_count,
        "total": total,
        "all_exact": exact_count == total and total > 0,
        "first_drift_indices": sorted(drift_indices) if drift_indices else [],
        "min_drift_index": min(drift_indices) if drift_indices else None,
        "ref_decode_tps": round(ref_tps, 2),
        "candidate_decode_tps": round(cand_tps, 2),
        "speedup": round(speedup, 4),
        "verdict": verdict,
    }


def generate_markdown(entries: list[dict], out_path: str) -> None:
    """Write Markdown summary table."""
    lines = []
    lines.append("# Qwen3.5-9B True-FP8 Layer-Mask Sweep Summary")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Reports analyzed: {len(entries)}")
    lines.append("")

    # Classification counts
    verdicts = {}
    for e in entries:
        v = e["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1
    lines.append("## Verdict Distribution")
    lines.append("")
    for v in ["EXACT_FAST", "EXACT_FLAT", "AMBER_LATE_DRIFT", "CLOSED_EARLY_DRIFT"]:
        if v in verdicts:
            lines.append(f"- **{v}**: {verdicts[v]}")
    lines.append("")

    # Table
    lines.append("## Results")
    lines.append("")
    lines.append("| Layer Mask | Exact | Ref TPS | Cand TPS | Speedup | Min Drift | Verdict |")
    lines.append("|------------|-------|---------|----------|---------|-----------|---------|")
    for e in sorted(entries, key=lambda x: x["verdict"]):
        mask_short = e["layer_mask"][:30]
        exact_str = f"{e['exact_count']}/{e['total']}"
        drift_str = str(e["min_drift_index"]) if e["min_drift_index"] is not None else "-"
        lines.append(
            f"| {mask_short} | {exact_str} | {e['ref_decode_tps']} | "
            f"{e['candidate_decode_tps']} | {e['speedup']:.3f} | "
            f"{drift_str} | {e['verdict']} |"
        )
    lines.append("")

    # Promotion rule
    lines.append("## Promotion Rule")
    lines.append("")
    lines.append("- EXACT_FAST (speedup >= 1.03): candidate for DEFAULT promotion")
    lines.append("- EXACT_FLAT (speedup < 1.03): exact but no speed gain, hold")
    lines.append("- AMBER_LATE_DRIFT (drift >= token 32): research, not promotable")
    lines.append("- CLOSED_EARLY_DRIFT (drift < token 32): reject")
    lines.append("")

    Path(out_path).write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize p190 true-FP8 layer-mask sweeps.")
    ap.add_argument("--reports", nargs="+", required=True,
                    help="Glob patterns or paths to p190 JSON reports")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    # Expand globs
    paths = []
    for pattern in args.reports:
        expanded = glob.glob(pattern)
        if expanded:
            paths.extend(expanded)
        elif os.path.exists(pattern):
            paths.append(pattern)

    if not paths:
        print("[p191] ERROR: no p190 reports found", file=sys.stderr)
        return 1

    print(f"[p191] Analyzing {len(paths)} report(s)...")

    entries = []
    for path in sorted(paths):
        try:
            with open(path) as f:
                report = json.load(f)
            entry = summarize_report(report, path)
            entries.append(entry)
            print(f"  {entry['layer_mask'][:25]:25s} {entry['exact_count']:2d}/{entry['total']:2d} "
                  f"spd={entry['speedup']:.3f} {entry['verdict']}")
        except Exception as e:
            print(f"  SKIP {os.path.basename(path)}: {e}", file=sys.stderr)

    if not entries:
        print("[p191] ERROR: no valid reports parsed", file=sys.stderr)
        return 1

    # Write JSON
    summary = {
        "schema": "lynn-qwen35-9b-true-fp8-layer-mask-summary-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_candidates": len(entries),
        "verdict_counts": {},
        "best_exact_fast": None,
        "entries": entries,
    }
    for e in entries:
        v = e["verdict"]
        summary["verdict_counts"][v] = summary["verdict_counts"].get(v, 0) + 1
    exact_fast = [e for e in entries if e["verdict"] == "EXACT_FAST"]
    if exact_fast:
        summary["best_exact_fast"] = max(exact_fast, key=lambda x: x["speedup"])

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    # Write Markdown
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    generate_markdown(entries, args.out_md)

    print(f"\n[p191] JSON: {args.out_json}")
    print(f"[p191] Markdown: {args.out_md}")
    print(f"[p191] Verdicts: {summary['verdict_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
