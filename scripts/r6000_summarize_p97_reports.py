#!/usr/bin/env python3
"""Summarize a batch of R6000 P97 interval reports.

This turns multiple per-layer P97 JSON reports into one compact summary so the
night run can leave behind a single handoff artifact.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_variant(row: dict[str, Any]) -> dict[str, Any]:
    variants = row.get("variants", [])
    if not variants:
        raise ValueError(f"missing variants in {row.get('path', '<memory>')}")
    best_name = row.get("best_variant")
    for variant in variants:
        if variant.get("name") == best_name:
            return variant
    return min(variants, key=lambda item: item.get("total_median_ms", float("inf")))


def summarize(paths: list[Path], model: str | None) -> dict[str, Any]:
    reports = [_read_json(path) for path in paths]
    rows = []
    for path, report in zip(paths, reports):
        best = _best_variant(report)
        diff = best["diff_vs_quantized_activation_active_reference"]
        rows.append(
            {
                "path": str(path),
                "layer": int(report["layer"]),
                "contract_pass": bool(report["contract_pass"]),
                "best_variant": best["name"],
                "baseline_total_median_ms": float(report["variants"][0]["total_median_ms"]),
                "best_total_median_ms": float(best["total_median_ms"]),
                "best_gate_median_ms": float(best["gate_median_ms"]),
                "best_down_median_ms": float(best["down_median_ms"]),
                "speedup": float(best["speedup_vs_baseline_total_median"]),
                "best_rel_l2": float(diff["rel_l2"]),
                "best_cosine": float(diff["cosine"]),
            }
        )

    rows.sort(key=lambda item: item["layer"])
    speedups = [row["speedup"] for row in rows]
    gate_ms = [row["best_gate_median_ms"] for row in rows]
    down_ms = [row["best_down_median_ms"] for row in rows]
    variants = sorted({row["best_variant"] for row in rows})
    all_contract_pass = all(row["contract_pass"] for row in rows)

    if all_contract_pass:
        if variants == ["p93_gateup_native_down_tile1"]:
            decision = (
                "native_down_tile1 repeatedly wins across sampled layers, but gate/up remains "
                "the dominant interval; pursue gate/up scheduling/fusion next before runtime promotion."
            )
        else:
            decision = "all sampled layers pass the quantized-activation contract; inspect the best variant mix before promotion."
    else:
        decision = "at least one sampled layer failed the quantized-activation contract; do not use the interval winner for runtime promotion."

    return {
        "schema_version": "r6000-v2-p97-multilayer-summary-v1",
        "model": model or reports[0].get("model"),
        "layers": [row["layer"] for row in rows],
        "all_contract_pass": all_contract_pass,
        "best_variant_set": variants,
        "speedup_min": min(speedups),
        "speedup_mean": float(statistics.fmean(speedups)),
        "speedup_max": max(speedups),
        "gate_median_mean_ms": float(statistics.fmean(gate_ms)),
        "down_median_mean_ms": float(statistics.fmean(down_ms)),
        "rows": rows,
        "decision": decision,
    }


def _select_paths(paths: list[Path], layers: set[int] | None, latest_per_layer: bool) -> list[Path]:
    selected: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        report = _read_json(path)
        layer = int(report["layer"])
        if layers is not None and layer not in layers:
            continue
        selected.append((path, report))

    if not latest_per_layer:
        return [path for path, _ in selected]

    latest: dict[int, tuple[Path, dict[str, Any], float]] = {}
    for path, report in selected:
        layer = int(report["layer"])
        mtime = path.stat().st_mtime
        prev = latest.get(layer)
        if prev is None or mtime > prev[2]:
            latest[layer] = (path, report, mtime)
    return [item[0] for _, item in sorted(latest.items())]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Glob for per-layer P97 JSON reports")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="Optional layer filter")
    ap.add_argument(
        "--latest-per-layer",
        action="store_true",
        help="When multiple reports exist for a layer, summarize only the newest one",
    )
    args = ap.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"no reports matched: {args.glob}")
    paths = _select_paths(paths, set(args.layers) if args.layers else None, args.latest_per_layer)
    if not paths:
        raise FileNotFoundError(f"no reports matched after filters: {args.glob}")

    result = summarize(paths, args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_contract_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
