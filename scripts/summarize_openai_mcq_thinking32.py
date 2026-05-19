#!/usr/bin/env python3
"""Summarize OpenAI-compatible 32K thinking MCQ JSONL outputs.

The evaluator writes one JSON object per question with fields such as
``gold``, ``pred``, ``ok``, ``subject``, ``raw_chars``, ``usage`` and
``elapsed_sec``.  This helper merges one or more JSONL files and emits a compact
JSON summary suitable for 9B/35B thinking-on quality gates.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


PARSER_VERSION = "thinking32-summary-v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                rows.append(
                    {
                        "_source": str(path),
                        "_line": line_no,
                        "error": f"JSONDecodeError: {exc}",
                        "pred": None,
                        "ok": False,
                    }
                )
                continue
            obj["_source"] = str(path)
            obj["_line"] = line_no
            rows.append(obj)
    return rows


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _usage_tokens(row: dict[str, Any], *keys: str) -> int | None:
    usage = row.get("usage") or {}
    if not isinstance(usage, dict):
        return None
    for key in keys:
        val = usage.get(key)
        if isinstance(val, int):
            return val
    return None


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(1 for row in rows if bool(row.get("ok")))
    parse_fail = sum(1 for row in rows if not row.get("pred"))
    errors = sum(1 for row in rows if row.get("error"))
    predicted = n - parse_fail
    raw_chars = [_number(row.get("raw_chars")) for row in rows]
    elapsed = [_number(row.get("elapsed_sec")) for row in rows]
    completion_tokens = [
        _usage_tokens(row, "completion_tokens", "output_tokens", "completion")
        for row in rows
    ]
    prompt_tokens = [
        _usage_tokens(row, "prompt_tokens", "input_tokens", "prompt")
        for row in rows
    ]
    total_tokens = [_usage_tokens(row, "total_tokens", "total") for row in rows]
    raw_chars_f = [x for x in raw_chars if x is not None]
    elapsed_f = [x for x in elapsed if x is not None]
    completion_f = [float(x) for x in completion_tokens if x is not None]
    prompt_f = [float(x) for x in prompt_tokens if x is not None]
    total_f = [float(x) for x in total_tokens if x is not None]
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "parse_fail": parse_fail,
        "parse_fail_rate": parse_fail / n if n else None,
        "accuracy_excluding_parse_fail": correct / predicted if predicted else None,
        "errors": errors,
        "raw_chars": _stats(raw_chars_f),
        "elapsed_sec": _stats(elapsed_f),
        "completion_tokens": _stats(completion_f),
        "prompt_tokens": _stats(prompt_f),
        "total_tokens": _stats(total_f),
    }


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = str(row.get(key) or "unknown")
        buckets[label].append(row)
    return {label: _bucket(items) for label, items in sorted(buckets.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+", help="One or more thinking32 JSONL files.")
    ap.add_argument("--out", required=True, help="Summary JSON output path.")
    ap.add_argument("--label", default="", help="Optional run label.")
    ap.add_argument(
        "--dedupe",
        action="store_true",
        help="Deduplicate by (source basename, id), keeping the last row.",
    )
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for item in args.jsonl:
        rows.extend(_read_jsonl(Path(item)))

    if args.dedupe:
        keyed: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (Path(str(row.get("_source", ""))).name, str(row.get("id") or row.get("_line")))
            keyed[key] = row
        rows = list(keyed.values())

    summary = {
        "schema": "lynn-openai-mcq-thinking32-summary-v1",
        "parser_version": PARSER_VERSION,
        "label": args.label,
        "sources": args.jsonl,
        "overall": _bucket(rows),
        "by_subject": _group_by(rows, "subject"),
        "by_source": _group_by(rows, "_source"),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
