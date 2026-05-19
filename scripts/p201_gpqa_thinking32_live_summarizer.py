#!/usr/bin/env python3
"""P201 · Live summarizer for in-progress GPQA Diamond thinking32 eval.

Reads a partial (still being written) JSONL from the R6000 thinking32 eval
and prints a compact live summary: accuracy, parse_fail, subject top-5 best
and worst, and per-subject breakdown for known hard domains.

Safe to run repeatedly while the eval is in progress — handles truncated
last lines, incomplete files, and growing data.

Usage:
  python scripts/p201_gpqa_thinking32_live_summarizer.py --jsonl JSONL_PATH
  python scripts/p201_gpqa_thinking32_live_summarizer.py JSONL_PATH --watch 30
  python scripts/p201_gpqa_thinking32_live_summarizer.py --out summary.json --md-out summary.md
  python scripts/p201_gpqa_thinking32_live_summarizer.py --report-dir reports/qwen35_9b
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_REPORT_DIR = Path("reports/qwen35_9b")
EXPECTED_GPQA_N = 198

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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _usage_token(row: dict[str, Any], *keys: str) -> float | None:
    usage = row.get("usage") or {}
    if not isinstance(usage, dict):
        return None
    for key in keys:
        value = usage.get(key)
        if _is_number(value):
            return float(value)
    return None


def _p95(values: list[int | float]) -> float:
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return float(ordered[idx])


def _num_summary(vals: list[int | float]) -> dict[str, float | int | None]:
    if not vals:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "p95": _p95(vals),
    }


def _fmt_number(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def _discover_jsonl(report_dir: Path) -> Path:
    patterns = [
        "thinking32/*thinking32*gpqa*.jsonl",
        "thinking32/*gpqa*thinking32*.jsonl",
        "*thinking32*gpqa*.jsonl",
        "*gpqa*thinking32*.jsonl",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in report_dir.glob(pattern) if path.is_file())
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(
            f"No GPQA thinking32 JSONL found under {report_dir}. "
            "Pass --jsonl /path/to/result.jsonl."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


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
        f"  Progress:    {n}/{EXPECTED_GPQA_N} questions",
        f"  Accuracy:    {correct}/{n} = {accuracy:.1%}",
        f"  Parse fail:  {parse_fail}/{n} = {parse_fail/n:.1%}" if n else "",
        f"  Acc (excl):  {correct}/{predicted} = {acc_excl:.1%}" if predicted != n else "",
        f"  Errors:      {errors}" if errors else "",
        f"",
    ]

    # Timing stats
    elapsed_vals = [r.get("elapsed_sec") for r in rows if _is_number(r.get("elapsed_sec"))]
    if elapsed_vals:
        avg_elapsed = sum(elapsed_vals) / len(elapsed_vals)
        med_elapsed = statistics.median(elapsed_vals)
        p95_elapsed = _p95(elapsed_vals)
        total_elapsed = sum(elapsed_vals)
        remaining = (EXPECTED_GPQA_N - n) * avg_elapsed if n < EXPECTED_GPQA_N else 0
        lines.append(f"  Avg time:    {avg_elapsed:.1f}s/question")
        lines.append(f"  Median/P95:  {med_elapsed:.1f}s / {p95_elapsed:.1f}s")
        lines.append(f"  Total time:  {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
        if remaining > 0:
            lines.append(f"  ETA remain:  ~{remaining:.0f}s ({remaining/60:.1f}min)")
        lines.append("")

    char_vals = [r.get("raw_chars") for r in rows if isinstance(r.get("raw_chars"), int)]
    completion_vals = [x for x in (_usage_token(r, "completion_tokens", "output_tokens", "completion") for r in rows) if x is not None]
    prompt_vals = [x for x in (_usage_token(r, "prompt_tokens", "input_tokens", "prompt") for r in rows) if x is not None]
    total_vals = [x for x in (_usage_token(r, "total_tokens", "total") for r in rows) if x is not None]
    if char_vals or prompt_vals or completion_vals or total_vals:
        if char_vals:
            lines.append(f"  Raw chars:   mean={sum(char_vals)/len(char_vals):.0f} median={statistics.median(char_vals):.0f}")
        if prompt_vals:
            lines.append(
                f"  Prompt tok:  mean={sum(prompt_vals)/len(prompt_vals):.0f} "
                f"median={statistics.median(prompt_vals):.0f}"
            )
        if completion_vals:
            lines.append(
                f"  Completion:  mean={sum(completion_vals)/len(completion_vals):.0f} "
                f"median={statistics.median(completion_vals):.0f} tokens"
            )
        if total_vals:
            lines.append(
                f"  Total tok:   mean={sum(total_vals)/len(total_vals):.0f} "
                f"median={statistics.median(total_vals):.0f}"
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
    elapsed_vals = [r.get("elapsed_sec") for r in rows if _is_number(r.get("elapsed_sec"))]
    raw_char_vals = [r.get("raw_chars") for r in rows if isinstance(r.get("raw_chars"), int)]
    prompt_vals = [x for x in (_usage_token(r, "prompt_tokens", "input_tokens", "prompt") for r in rows) if x is not None]
    completion_vals = [x for x in (_usage_token(r, "completion_tokens", "output_tokens", "completion") for r in rows) if x is not None]
    total_vals = [x for x in (_usage_token(r, "total_tokens", "total") for r in rows) if x is not None]

    return {
        "schema": "lynn-p201-gpqa-thinking32-summary-v2",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": jsonl_path,
        "progress": f"{n}/{EXPECTED_GPQA_N}",
        "overall": {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else None,
            "parse_fail": parse_fail,
            "parse_fail_rate": parse_fail / n if n else None,
            "excl_parse_fail": correct / predicted if predicted else None,
            "accuracy_excluding_parse_fail": correct / predicted if predicted else None,
        },
        "elapsed_sec": _num_summary(elapsed_vals),
        "raw_chars": _num_summary(raw_char_vals),
        "tokens": {
            "prompt": _num_summary(prompt_vals),
            "completion": _num_summary(completion_vals),
            "total": _num_summary(total_vals),
        },
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


def _markdown_summary(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    elapsed = summary["elapsed_sec"]
    tokens = summary["tokens"]
    lines = [
        "# P201 Qwen3.5-9B Thinking32 GPQA Summary",
        "",
        f"- Source: `{summary['source']}`",
        f"- Progress: {summary['progress']}",
        f"- Accuracy: {_fmt_number(overall['accuracy'] * 100 if overall['accuracy'] is not None else None, '%')} "
        f"({overall['correct']}/{overall['n']})",
        f"- Parse fail: {overall['parse_fail']} "
        f"({_fmt_number(overall['parse_fail_rate'] * 100 if overall['parse_fail_rate'] is not None else None, '%')})",
        f"- Excl parse fail: {_fmt_number(overall['excl_parse_fail'] * 100 if overall['excl_parse_fail'] is not None else None, '%')}",
        f"- Elapsed seconds: avg {_fmt_number(elapsed['mean'])}, median {_fmt_number(elapsed['median'])}, p95 {_fmt_number(elapsed['p95'])}",
        f"- Prompt tokens: avg {_fmt_number(tokens['prompt']['mean'])}, median {_fmt_number(tokens['prompt']['median'])}, p95 {_fmt_number(tokens['prompt']['p95'])}",
        f"- Completion tokens: avg {_fmt_number(tokens['completion']['mean'])}, median {_fmt_number(tokens['completion']['median'])}, p95 {_fmt_number(tokens['completion']['p95'])}",
        f"- Total tokens: avg {_fmt_number(tokens['total']['mean'])}, median {_fmt_number(tokens['total']['median'])}, p95 {_fmt_number(tokens['total']['p95'])}",
        "",
    ]
    if summary["top5_parse_fail"]:
        lines.extend(["## Parse Fail Leaders", ""])
        for item in summary["top5_parse_fail"]:
            lines.append(f"- {item['subject']}: {item['parse_fail']}/{item['n']}")
        lines.append("")
    return "\n".join(lines)


def _default_output_paths(jsonl_path: Path, report_dir: Path) -> tuple[Path, Path]:
    stem = jsonl_path.stem
    out_dir = report_dir if report_dir.exists() else jsonl_path.parent
    return (
        out_dir / f"{stem}.p201_summary.json",
        out_dir / f"{stem}.p201_summary.md",
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="P201: Live summary of in-progress GPQA Diamond thinking32 eval."
    )
    ap.add_argument("jsonl_arg", nargs="?", help="Legacy positional JSONL path.")
    ap.add_argument("--jsonl", help="Path to the (possibly partial) JSONL output file.")
    ap.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Report directory used for auto-discovery and default outputs.",
    )
    ap.add_argument("--out", default=None, help="Write JSON summary to this path.")
    ap.add_argument("--md-out", default=None, help="Write Markdown summary to this path.")
    ap.add_argument(
        "--no-default-out",
        action="store_true",
        help="Do not write default JSON/Markdown outputs when --out/--md-out are omitted.",
    )
    ap.add_argument(
        "--watch", type=int, default=0, metavar="SECS",
        help="Re-read and reprint every N seconds (0=once, exit).",
    )
    ap.add_argument("--json", action="store_true", help="Print JSON instead of table.")
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    jsonl_path = Path(args.jsonl or args.jsonl_arg) if (args.jsonl or args.jsonl_arg) else _discover_jsonl(report_dir)
    default_json, default_md = _default_output_paths(jsonl_path, report_dir)
    out_path = Path(args.out) if args.out else (None if args.no_default_out else default_json)
    md_path = Path(args.md_out) if args.md_out else (None if args.no_default_out else default_md)

    while True:
        rows = _read_partial_jsonl(jsonl_path)
        summary = _json_summary(rows, str(jsonl_path))

        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            text = _format_summary(rows, str(jsonl_path))
            # Clear screen for watch mode
            if args.watch > 0:
                print("\033[2J\033[H", end="")
            print(text)

        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if md_path:
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(_markdown_summary(summary) + "\n", encoding="utf-8")

        if args.watch <= 0:
            break
        time.sleep(args.watch)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
