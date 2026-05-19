#!/usr/bin/env python3
"""P201 · Live summarizer for in-progress GPQA Diamond thinking32 eval.

Reads a partial (still being written) JSONL from the R6000 thinking32 eval
and prints a compact live summary: accuracy, parse_fail, subject top-5 best
and worst, and per-subject breakdown for known hard domains.

Safe to run repeatedly while the eval is in progress — handles truncated
last lines, incomplete files, and growing data.

Usage:
  python scripts/p201_gpqa_thinking32_live_summarizer.py JSONL_PATH
  python scripts/p201_gpqa_thinking32_live_summarizer.py JSONL_PATH --watch 30
  python scripts/p201_gpqa_thinking32_live_summarizer.py JSONL_PATH --out summary.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


# Known hard subjects in GPQA Diamond (for special attention)
HARD_SUBJECTS = [
    "Organic Chemistry",
    "Chemistry (general)",
    "Quantum Mechanics",
    "Molecular Biology",
    "Genetics",
    "Condensed Matter Physics",
    "Astrophysics",
    "High Energy Physics",
]


def _read_partial_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL, tolerating truncated last line (eval still writing).

    Normalizes two JSONL schemas:
      - thinking32: {id, subject, gold, pred, ok, elapsed_sec, raw_chars, usage}
      - gpqa_eval:  {id, answer, prediction, correct, response, usage, error}
    Into: {id, subject, gold, pred, ok, elapsed_sec, raw_chars, usage, error}
    """
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                # Truncated last line — eval still writing. Skip silently.
                continue
            # Normalize field names
            obj["_line"] = line_no
            if "pred" not in obj and "prediction" in obj:
                obj["pred"] = obj["prediction"]
            if "ok" not in obj and "correct" in obj:
                obj["ok"] = obj["correct"]
            if "gold" not in obj and "answer" in obj:
                obj["gold"] = obj["answer"]
            if "subject" not in obj:
                obj["subject"] = obj.get("Subdomain") or obj.get("High-level domain") or ""
            rows.append(obj)
    return rows


def _subject_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group by subject, compute per-subject stats."""
    by_subj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subj = (row.get("subject") or "unknown").strip()
        if not subj:
            subj = "unknown"
        by_subj[subj].append(row)

    result = {}
    for subj, items in sorted(by_subj.items()):
        n = len(items)
        correct = sum(1 for r in items if r.get("ok"))
        parse_fail = sum(1 for r in items if not r.get("pred"))
        predicted = n - parse_fail
        result[subj] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else 0.0,
            "accuracy_excl_pf": correct / predicted if predicted else 0.0,
            "parse_fail": parse_fail,
        }
    return result


def _format_summary(rows: list[dict[str, Any]], jsonl_path: str) -> str:
    """Format a human-readable live summary."""
    n = len(rows)
    if n == 0:
        return f"[P201] {jsonl_path}: 0 rows (file empty or not yet started)\n"

    correct = sum(1 for r in rows if r.get("ok"))
    parse_fail = sum(1 for r in rows if not r.get("pred"))
    errors = sum(1 for r in rows if r.get("error"))
    predicted = n - parse_fail
    accuracy = correct / n if n else 0.0
    acc_excl = correct / predicted if predicted else 0.0

    lines = [
        f"╔══════════════════════════════════════════════════════════════╗",
        f"║  P201 · GPQA Diamond Thinking32 Live Summary                ║",
        f"╚══════════════════════════════════════════════════════════════╝",
        f"",
        f"  Source:      {jsonl_path}",
        f"  Progress:    {n}/198 questions",
        f"  Accuracy:    {correct}/{n} = {accuracy:.1%}",
        f"  Parse fail:  {parse_fail}/{n} = {parse_fail/n:.1%}" if n else "",
        f"  Acc (excl):  {correct}/{predicted} = {acc_excl:.1%}" if predicted != n else "",
        f"  Errors:      {errors}" if errors else "",
        f"",
    ]

    # Timing stats
    elapsed_vals = [r.get("elapsed_sec") for r in rows if isinstance(r.get("elapsed_sec"), (int, float))]
    if elapsed_vals:
        avg_elapsed = sum(elapsed_vals) / len(elapsed_vals)
        med_elapsed = statistics.median(elapsed_vals)
        p95_elapsed = sorted(elapsed_vals)[min(len(elapsed_vals) - 1, int(len(elapsed_vals) * 0.95))]
        total_elapsed = sum(elapsed_vals)
        remaining = (198 - n) * avg_elapsed if n < 198 else 0
        lines.append(f"  Avg time:    {avg_elapsed:.1f}s/question")
        lines.append(f"  Median/P95:  {med_elapsed:.1f}s / {p95_elapsed:.1f}s")
        lines.append(f"  Total time:  {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
        if remaining > 0:
            lines.append(f"  ETA remain:  ~{remaining:.0f}s ({remaining/60:.1f}min)")
        lines.append("")

    char_vals = [r.get("raw_chars") for r in rows if isinstance(r.get("raw_chars"), int)]
    completion_vals = [
        (r.get("usage") or {}).get("completion_tokens")
        for r in rows
        if isinstance((r.get("usage") or {}).get("completion_tokens"), int)
    ]
    if char_vals or completion_vals:
        if char_vals:
            lines.append(f"  Raw chars:   mean={sum(char_vals)/len(char_vals):.0f} median={statistics.median(char_vals):.0f}")
        if completion_vals:
            lines.append(
                f"  Completion:  mean={sum(completion_vals)/len(completion_vals):.0f} "
                f"median={statistics.median(completion_vals):.0f} tokens"
            )
        lines.append("")

    # Subject breakdown
    subj_stats = _subject_stats(rows)

    if subj_stats:
        # Top 5 best accuracy (with n >= 2)
        ranked = sorted(
            [(s, v) for s, v in subj_stats.items() if v["n"] >= 2],
            key=lambda x: x[1]["accuracy"],
            reverse=True,
        )
        lines.append("  ─── Top 5 Subjects (best accuracy) ───")
        for subj, v in ranked[:5]:
            lines.append(
                f"    {subj:35s} {v['correct']}/{v['n']} = {v['accuracy']:.0%}"
                + (f"  (pf={v['parse_fail']})" if v["parse_fail"] else "")
            )
        lines.append("")

        # Top 5 worst accuracy (with n >= 2)
        worst = sorted(
            [(s, v) for s, v in subj_stats.items() if v["n"] >= 2],
            key=lambda x: x[1]["accuracy"],
        )
        lines.append("  ─── Top 5 Subjects (worst accuracy) ───")
        for subj, v in worst[:5]:
            lines.append(
                f"    {subj:35s} {v['correct']}/{v['n']} = {v['accuracy']:.0%}"
                + (f"  (pf={v['parse_fail']})" if v["parse_fail"] else "")
            )
        lines.append("")

        # Parse fail leaders
        pf_ranked = sorted(
            [(s, v) for s, v in subj_stats.items() if v["parse_fail"] > 0],
            key=lambda x: x[1]["parse_fail"],
            reverse=True,
        )
        if pf_ranked:
            lines.append("  ─── Parse Fail by Subject (top 5) ───")
            for subj, v in pf_ranked[:5]:
                lines.append(
                    f"    {subj:35s} pf={v['parse_fail']}/{v['n']}"
                )
            lines.append("")

        # Hard subjects detail
        hard_present = [(s, subj_stats[s]) for s in HARD_SUBJECTS if s in subj_stats]
        if hard_present:
            lines.append("  ─── Hard Subject Detail ───")
            for subj, v in hard_present:
                bar = "█" * v["correct"] + "░" * (v["n"] - v["correct"])
                lines.append(
                    f"    {subj:30s} {v['correct']}/{v['n']} = {v['accuracy']:.0%} [{bar}]"
                    + (f" pf={v['parse_fail']}" if v["parse_fail"] else "")
                )
            lines.append("")

    # Full subject table (compact)
    if subj_stats and len(subj_stats) <= 30:
        lines.append("  ─── Full Subject Table ───")
        lines.append(f"    {'Subject':35s} {'Acc':>6s} {'N':>3s} {'PF':>3s}")
        lines.append(f"    {'─' * 35} {'─' * 6} {'─' * 3} {'─' * 3}")
        for subj, v in sorted(subj_stats.items(), key=lambda x: x[1]["accuracy"]):
            lines.append(
                f"    {subj:35s} {v['accuracy']:5.0%} {v['n']:3d} {v['parse_fail']:3d}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _json_summary(rows: list[dict[str, Any]], jsonl_path: str) -> dict[str, Any]:
    """Produce a machine-readable summary dict."""
    n = len(rows)
    correct = sum(1 for r in rows if r.get("ok"))
    parse_fail = sum(1 for r in rows if not r.get("pred"))
    predicted = n - parse_fail

    subj_stats = _subject_stats(rows)
    ranked = sorted(
        [(s, v) for s, v in subj_stats.items() if v["n"] >= 2],
        key=lambda x: x[1]["accuracy"],
        reverse=True,
    )
    worst = sorted(
        [(s, v) for s, v in subj_stats.items() if v["n"] >= 2],
        key=lambda x: x[1]["accuracy"],
    )
    pf_ranked = sorted(
        [(s, v) for s, v in subj_stats.items() if v["parse_fail"] > 0],
        key=lambda x: x[1]["parse_fail"],
        reverse=True,
    )
    elapsed_vals = [r.get("elapsed_sec") for r in rows if isinstance(r.get("elapsed_sec"), (int, float))]
    raw_char_vals = [r.get("raw_chars") for r in rows if isinstance(r.get("raw_chars"), int)]
    completion_vals = [
        (r.get("usage") or {}).get("completion_tokens")
        for r in rows
        if isinstance((r.get("usage") or {}).get("completion_tokens"), int)
    ]

    def _num_summary(vals: list[int | float]) -> dict[str, float | None]:
        if not vals:
            return {"mean": None, "median": None, "p95": None}
        ordered = sorted(vals)
        return {
            "mean": sum(vals) / len(vals),
            "median": statistics.median(vals),
            "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        }

    return {
        "schema": "lynn-p201-gpqa-thinking32-live-summary-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": jsonl_path,
        "progress": f"{n}/198",
        "overall": {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else None,
            "parse_fail": parse_fail,
            "parse_fail_rate": parse_fail / n if n else None,
            "accuracy_excl_pf": correct / predicted if predicted else None,
        },
        "timing": _num_summary(elapsed_vals),
        "raw_chars": _num_summary(raw_char_vals),
        "completion_tokens": _num_summary(completion_vals),
        "top5_best": [
            {"subject": s, **v} for s, v in ranked[:5]
        ],
        "top5_worst": [
            {"subject": s, **v} for s, v in worst[:5]
        ],
        "top5_parse_fail": [
            {"subject": s, **v} for s, v in pf_ranked[:5]
        ],
        "hard_subjects": {
            s: subj_stats[s] for s in HARD_SUBJECTS if s in subj_stats
        },
        "by_subject": subj_stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="P201: Live summary of in-progress GPQA Diamond thinking32 eval."
    )
    ap.add_argument("jsonl", help="Path to the (possibly partial) JSONL output file.")
    ap.add_argument("--out", default=None, help="Write JSON summary to this path.")
    ap.add_argument(
        "--watch", type=int, default=0, metavar="SECS",
        help="Re-read and reprint every N seconds (0=once, exit).",
    )
    ap.add_argument("--json", action="store_true", help="Print JSON instead of table.")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)

    while True:
        rows = _read_partial_jsonl(jsonl_path)

        if args.json:
            summary = _json_summary(rows, str(jsonl_path))
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            text = _format_summary(rows, str(jsonl_path))
            # Clear screen for watch mode
            if args.watch > 0:
                print("\033[2J\033[H", end="")
            print(text)

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            summary = _json_summary(rows, str(jsonl_path))
            out_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        if args.watch <= 0:
            break
        time.sleep(args.watch)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
