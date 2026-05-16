#!/usr/bin/env python3
"""Rank W4A8 Recovery generation-gate candidates.

The A100 loop can produce many small Recovery candidates. This script keeps the
decision rule boring and repeatable: exact-match first, then worst-case shared
prefix, then mean shared prefix. It also highlights high-risk divergences so a
candidate with late paraphrases does not get treated the same as one that breaks
JSON/tool-call behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HIGH_RISK_MARKERS = (
    "{",
    "}",
    '"tool"',
    "arguments",
    "JSON",
    "json",
    "```",
    "<think>",
    "</think>",
)


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_name(path: Path, data: dict[str, Any]) -> str:
    models = data.get("models", {})
    candidate = models.get("candidate") if isinstance(models, dict) else None
    if candidate:
        return Path(str(candidate)).name
    stem = path.stem
    prefix = "a100_w4a8_generation_gate_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def _risk(row: dict[str, Any]) -> str:
    if row.get("exact"):
        return "exact"
    prompt = str(row.get("prompt", ""))
    candidate = str(row.get("candidate_full_text", ""))
    reference = str(row.get("reference_off_text", ""))
    first_diff = int(row.get("first_diff_index") or 0)
    if any(marker in prompt or marker in candidate or marker in reference for marker in HIGH_RISK_MARKERS):
        return "high"
    if first_diff <= 8:
        return "high"
    if first_diff <= 24:
        return "medium"
    return "low"


def _score(path: Path) -> dict[str, Any]:
    data = _load(path)
    cmp = data.get("cross_model_compare", {})
    rows = list(cmp.get("rows", []))
    total = len(rows)
    exact = int(cmp.get("exact", sum(1 for r in rows if r.get("exact"))))
    min_prefix = float(cmp.get("min_same_prefix_tokens", 0.0))
    mean_prefix = float(cmp.get("mean_same_prefix_tokens", 0.0))
    risks = {"high": 0, "medium": 0, "low": 0}
    worst_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("exact"):
            continue
        risk = _risk(row)
        risks[risk] += 1
        worst_rows.append(
            {
                "prompt_id": row.get("prompt_id"),
                "risk": risk,
                "same_prefix_tokens": row.get("same_prefix_tokens"),
                "prompt": row.get("prompt"),
            }
        )
    worst_rows.sort(key=lambda r: (r["risk"] != "high", r["same_prefix_tokens"] or 10**9))
    return {
        "path": str(path),
        "candidate": _candidate_name(path, data),
        "decision": data.get("decision", ""),
        "exact": exact,
        "total": total,
        "min_prefix": min_prefix,
        "mean_prefix": mean_prefix,
        "risk_counts": risks,
        "worst_rows": worst_rows[:5],
    }


def _sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item["exact"],
        item["min_prefix"],
        item["mean_prefix"],
        -item["risk_counts"]["high"],
        -item["risk_counts"]["medium"],
    )


def _print_table(items: list[dict[str, Any]]) -> None:
    print("| rank | candidate | exact | min_prefix | mean_prefix | high/med/low | decision |")
    print("|---:|---|---:|---:|---:|---:|---|")
    for idx, item in enumerate(items, 1):
        risks = item["risk_counts"]
        print(
            f"| {idx} | {item['candidate']} | {item['exact']}/{item['total']} | "
            f"{item['min_prefix']:.0f} | {item['mean_prefix']:.2f} | "
            f"{risks['high']}/{risks['medium']}/{risks['low']} | {item['decision']} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Generation gate JSON files or globs.")
    parser.add_argument("--json-out", help="Optional path for machine-readable ranking.")
    args = parser.parse_args()

    paths: list[Path] = []
    for value in args.paths:
        matches = sorted(Path().glob(value))
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(value))

    items = [_score(path) for path in paths if path.exists()]
    items.sort(key=_sort_key, reverse=True)
    _print_table(items)

    if items:
        champion = items[0]
        print()
        print(
            "champion="
            f"{champion['candidate']} exact={champion['exact']}/{champion['total']} "
            f"min_prefix={champion['min_prefix']:.0f} mean_prefix={champion['mean_prefix']:.2f}"
        )
        if champion["worst_rows"]:
            print("worst_divergences:")
            for row in champion["worst_rows"]:
                print(
                    f"- prompt_id={row['prompt_id']} risk={row['risk']} "
                    f"prefix={row['same_prefix_tokens']} prompt={row['prompt']!r}"
                )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
