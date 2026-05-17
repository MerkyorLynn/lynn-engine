#!/usr/bin/env python3
"""Summarize A100 iterative MTP training reports into a compact ladder."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(report: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = report.get(key)
    if not isinstance(value, dict):
        return None
    return {
        "accepted": value.get("accepted"),
        "total": value.get("case_count"),
        "accept_rate": value.get("accept_rate"),
        "mean_loss": value.get("mean_loss"),
        "by_step": value.get("by_step", {}),
    }


def summarize(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        report = _read(path)
        eval_before = _metric(report, "eval_before")
        eval_after = _metric(report, "eval_after")
        train_before = _metric(report, "train_before")
        train_after = _metric(report, "train_after")
        rows.append(
            {
                "path": str(path),
                "trainable": report.get("trainable"),
                "steps": report.get("steps"),
                "lr": report.get("lr"),
                "first_token_weight": report.get("first_token_weight"),
                "step1_weight": report.get("step1_weight"),
                "later_token_weight": report.get("later_token_weight"),
                "max_new_train": report.get("max_new_train"),
                "max_new_eval": report.get("max_new_eval"),
                "source_sidecar_file": report.get("source_sidecar_file"),
                "trained_sidecar_file": report.get("trained_sidecar_file"),
                "train_before": train_before,
                "train_after": train_after,
                "eval_before": eval_before,
                "eval_after": eval_after,
                "eval_accept_delta": (
                    None
                    if not eval_before or not eval_after
                    else float(eval_after["accept_rate"] - eval_before["accept_rate"])
                ),
            }
        )
    rows.sort(key=lambda row: _timestamp_key(Path(row["path"])))
    best = max(
        rows,
        key=lambda row: -1.0 if row.get("eval_after") is None else float(row["eval_after"]["accept_rate"]),
        default=None,
    )
    return {
        "schema_version": "lynn-a100-mtp-iterative-ladder-summary-v1",
        "report_count": len(rows),
        "best_eval_after_path": None if best is None else best["path"],
        "best_eval_after_accept_rate": None if best is None else best["eval_after"]["accept_rate"],
        "rows": rows,
        "decision": (
            "No iterative reports found."
            if not rows
            else "MTP iterative accept is improving but remains below serving multiplier threshold."
        ),
    }


def _timestamp_key(path: Path) -> tuple[str, str]:
    match = re.search(r"(20\d{6}_\d{6})", path.name)
    return (match.group(1) if match else "", path.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted(Path(path) for path in glob.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"no reports matched: {args.glob}")
    result = summarize(paths)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
