#!/usr/bin/env python3
"""Summarize R6000 OpenAI server decode sweeps.

The env sweep reports are intentionally close to the raw benchmark output. This
helper turns one or more sweep JSON files into a compact comparison table so a
runtime candidate can be promoted or rejected without hand-editing jq snippets.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_rows(path: Path, report: dict[str, Any], baseline: str) -> list[dict[str, Any]]:
    tps_by_config = report.get("decode_tps_mean_by_config", {})
    flags_by_config = report.get("quality_flags", {})
    baseline_by_tokens = tps_by_config.get(baseline, {})
    rows: list[dict[str, Any]] = []
    for config, by_tokens in sorted(tps_by_config.items()):
        for token_count, decode_tps in sorted(by_tokens.items(), key=lambda item: int(item[0])):
            base_tps = baseline_by_tokens.get(token_count)
            speedup = None
            if base_tps and decode_tps:
                speedup = float(decode_tps) / float(base_tps)
            flags = flags_by_config.get(config, {})
            rows.append(
                {
                    "sweep": str(path),
                    "timestamp": report.get("timestamp"),
                    "config": config,
                    "max_tokens": int(token_count),
                    "decode_tps": decode_tps,
                    "baseline_config": baseline,
                    "baseline_decode_tps": base_tps,
                    "speedup_vs_baseline": speedup,
                    "all_preview_exclamation_loop": flags.get("all_preview_exclamation_loop"),
                }
            )
    return rows


def summarize(paths: list[Path], baseline: str, min_speedup: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_candidate_rows(path, _read(path), baseline))

    candidate_rows = [row for row in rows if row["config"] != baseline]
    promoted = [
        row
        for row in candidate_rows
        if row.get("speedup_vs_baseline") is not None
        and float(row["speedup_vs_baseline"]) >= min_speedup
        and not row.get("all_preview_exclamation_loop")
    ]

    best = None
    if candidate_rows:
        best = max(
            candidate_rows,
            key=lambda row: -1.0
            if row.get("speedup_vs_baseline") is None
            else float(row["speedup_vs_baseline"]),
        )

    return {
        "schema_version": "r6000-server-decode-sweep-summary-v1",
        "baseline_config": baseline,
        "min_promote_speedup": min_speedup,
        "sweep_count": len(paths),
        "rows": rows,
        "best_candidate": best,
        "decision": (
            f"GREEN: at least one candidate cleared {min_speedup:.3f}x without preview loop."
            if promoted
            else f"RED: no candidate cleared {min_speedup:.3f}x without preview loop."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Glob for r6000_p25_server_decode_sweep_*.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="configD")
    ap.add_argument("--min-speedup", type=float, default=1.03)
    args = ap.parse_args()

    paths = sorted(Path(path) for path in glob.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"no sweep reports matched: {args.glob}")

    result = summarize(paths, args.baseline, args.min_speedup)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
