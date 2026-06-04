#!/usr/bin/env python3
"""Build a canonical R5-C4 candidate JSON from raw harness metrics.

The R5-C4 validator intentionally consumes a narrow candidate schema. This
adapter lets a real R6000 kernel harness emit simpler raw metrics while keeping
the promotion boundary strict and reproducible.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "lynn-stage6-r5c4-candidate-v1"
REQUIRED_LANES = ("smoke", "production_shape")
PASS_KEYS = (
    "same_scope_ab",
    "real_model_weights",
    "real_router_outputs",
    "candidate_no_active_bf16_shadow",
    "candidate_no_reload",
    "candidate_no_bf16_weight_materialization",
    "candidate_full_active_moe_boundary_timed",
    "timing_includes_gateup_swiglu_down_weighted_scatter",
)
LANE_BOOL_KEYS = ("route_order_preserved", "repack_cost_reported", "fault_injections_detected")
FORBIDDEN_KEYS = (
    "banked_decode_tps",
    "banked_server_rc",
    "banked_default_promotion",
    "banked_full_transformer_prefill",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected top-level JSON object")
    return data


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _as_float(raw: dict[str, Any], key: str, lane_name: str, failures: list[str]) -> float:
    value = raw.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        failures.append(f"lane {lane_name}: {key} must be numeric")
        return float("nan")
    if not math.isfinite(number):
        failures.append(f"lane {lane_name}: {key} must be finite")
    return number


def _as_bool(raw: dict[str, Any], key: str, failures: list[str], *, context: str) -> bool:
    value = raw.get(key)
    if value is not True:
        failures.append(f"{context}: {key} must be true")
    return value is True


def _shape(raw: dict[str, Any]) -> dict[str, Any]:
    shape = raw.get("shape")
    if isinstance(shape, dict):
        return shape
    keys = ("tokens", "hidden", "intermediate", "top_k", "experts", "active_experts")
    return {key: raw[key] for key in keys if key in raw}


def _normalize_lane(name: str, raw: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        failures.append(f"lane {name}: expected object")
        return {}
    lane: dict[str, Any] = {
        "shape": _shape(raw),
        "candidate_ms": _as_float(raw, "candidate_ms", name, failures),
        "baseline_w4a16_ms": _as_float(raw, "baseline_w4a16_ms", name, failures),
        "baseline_packed_p3_ms": None,
        "numeric_max_abs": _as_float(raw, "numeric_max_abs", name, failures),
        "numeric_rel_l2": _as_float(raw, "numeric_rel_l2", name, failures),
        "numeric_cosine": _as_float(raw, "numeric_cosine", name, failures),
        "route_order_preserved": raw.get("route_order_preserved") is True,
        "repack_cost_reported": raw.get("repack_cost_reported") is True,
        "fault_injections_detected": raw.get("fault_injections_detected") is True,
        "shape_regression": raw.get("shape_regression") is True,
    }
    p3_value = raw.get("baseline_packed_p3_ms")
    if p3_value is not None:
        try:
            p3_ms = float(p3_value)
        except (TypeError, ValueError):
            failures.append(f"lane {name}: baseline_packed_p3_ms must be numeric when present")
        else:
            lane["baseline_packed_p3_ms"] = p3_ms if math.isfinite(p3_ms) and p3_ms > 0.0 else None
    if not _finite_positive(lane["candidate_ms"]):
        failures.append(f"lane {name}: candidate_ms must be > 0")
    if not _finite_positive(lane["baseline_w4a16_ms"]):
        failures.append(f"lane {name}: baseline_w4a16_ms must be > 0")
    for key in LANE_BOOL_KEYS:
        if raw.get(key) is not True:
            failures.append(f"lane {name}: {key} must be true")
    if not lane["shape"]:
        failures.append(f"lane {name}: shape must be reported")
    return lane


def build_candidate(raw: dict[str, Any], *, source: Path | None = None) -> dict[str, Any]:
    failures: list[str] = []
    raw_passes = raw.get("passes") if isinstance(raw.get("passes"), dict) else raw
    for key in FORBIDDEN_KEYS:
        if raw.get(key) is True or (isinstance(raw.get("passes"), dict) and raw["passes"].get(key) is True):
            failures.append(f"forbidden promotion field must not be true: {key}")
    passes = {
        key: _as_bool(raw_passes, key, failures, context="passes")
        for key in PASS_KEYS
    }
    raw_lanes = raw.get("lanes")
    if not isinstance(raw_lanes, dict):
        failures.append("lanes must be an object containing smoke and production_shape")
        raw_lanes = {}
    lanes = {
        name: _normalize_lane(name, raw_lanes.get(name, {}), failures)
        for name in REQUIRED_LANES
    }
    extra_lanes = {
        name: _normalize_lane(name, lane, failures)
        for name, lane in sorted(raw_lanes.items())
        if name not in REQUIRED_LANES
    }
    lanes.update(extra_lanes)
    candidate = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_raw_metrics": str(source) if source is not None else raw.get("source_raw_metrics"),
        "candidate_name": raw.get("candidate_name") or raw.get("candidate") or "unnamed-r5c4-candidate",
        "implementation": raw.get("implementation", {}),
        "passes": passes,
        "lanes": lanes,
        "promotion_boundary_request": {
            "decode_tps": False,
            "server_rc": False,
            "default_runtime": False,
            "full_transformer_prefill": False,
        },
    }
    if failures:
        candidate["adapter_failures"] = failures
    return candidate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-metrics", required=True, help="Raw R6000 harness metrics JSON")
    ap.add_argument("--out", required=True, help="Canonical candidate JSON output")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if required fields are missing")
    args = ap.parse_args()
    raw_path = Path(args.raw_metrics).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    candidate = build_candidate(_load_json(raw_path), source=raw_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures = candidate.get("adapter_failures", [])
    print(json.dumps({"schema": candidate["schema"], "adapter_failures": failures}, indent=2))
    return 2 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
