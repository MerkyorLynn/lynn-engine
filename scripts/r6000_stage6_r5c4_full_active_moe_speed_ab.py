#!/usr/bin/env python3
"""Stage 6 R5-C4 full active-MoE prefill speed A/B validator.

This is the R6000-side evidence validator for a future full active-MoE candidate.
It does not implement the kernel. It consumes a candidate JSON produced by the
kernel harness, verifies the same-scope numeric/timing contract, and emits the
canonical R5-C4 PASS/DIAGNOSTIC/FAIL artifact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


PASS_DECISION = "PASS_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB"
DIAGNOSTIC_DECISION = "DIAGNOSTIC_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_CLOSED"
FAIL_DECISION = "FAIL_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB"
DEFAULT_R5C3C = (
    "reports/stage6/"
    "r5c3c_down_weighted_parity_smoke_20260604_130243/result.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(data: dict[str, Any], key: str) -> bool:
    return data.get(key) is True


def _as_float(data: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def _finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _lane_report(name: str, raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    candidate_ms = _as_float(raw, "candidate_ms")
    w4a16_ms = _as_float(raw, "baseline_w4a16_ms")
    p3_ms = _as_float(raw, "baseline_packed_p3_ms")
    references = [value for value in [w4a16_ms, p3_ms] if _finite_positive(value)]
    best_reference_ms = min(references) if references else float("nan")
    speedup_vs_w4a16 = w4a16_ms / candidate_ms if _finite_positive(w4a16_ms) and _finite_positive(candidate_ms) else float("nan")
    speedup_vs_packed_p3 = p3_ms / candidate_ms if _finite_positive(p3_ms) and _finite_positive(candidate_ms) else float("nan")
    speedup_vs_best = best_reference_ms / candidate_ms if _finite_positive(best_reference_ms) and _finite_positive(candidate_ms) else float("nan")
    numeric_max_abs = _as_float(raw, "numeric_max_abs")
    numeric_rel_l2 = _as_float(raw, "numeric_rel_l2")
    numeric_cosine = _as_float(raw, "numeric_cosine")
    numeric_ok = (
        math.isfinite(numeric_max_abs)
        and math.isfinite(numeric_rel_l2)
        and math.isfinite(numeric_cosine)
        and numeric_max_abs <= args.max_abs_threshold
        and numeric_rel_l2 <= args.rel_l2_threshold
        and numeric_cosine >= args.cosine_threshold
    )
    p3_speed_ok = True if not _finite_positive(p3_ms) else speedup_vs_packed_p3 >= 1.0
    passes = {
        "candidate_ms_positive": _finite_positive(candidate_ms),
        "w4a16_baseline_ms_positive": _finite_positive(w4a16_ms),
        "best_reference_ms_positive": _finite_positive(best_reference_ms),
        "numeric_thresholds_passed": numeric_ok,
        "route_order_preserved": _as_bool(raw, "route_order_preserved"),
        "repack_cost_reported": _as_bool(raw, "repack_cost_reported"),
        "fault_injections_detected": _as_bool(raw, "fault_injections_detected"),
        "speedup_vs_w4a16_ge_1p10": speedup_vs_w4a16 >= 1.10,
        "speedup_vs_packed_p3_ge_1p00_or_unavailable": p3_speed_ok,
        "median_speedup_vs_best_reference_ge_1p05": speedup_vs_best >= 1.05,
        "no_shape_regression": raw.get("shape_regression") is not True,
    }
    passes["numeric_parity_passed"] = all(
        bool(passes[key])
        for key in [
            "numeric_thresholds_passed",
            "route_order_preserved",
            "repack_cost_reported",
            "fault_injections_detected",
        ]
    )
    passes["speed_gate_passed"] = all(
        bool(passes[key])
        for key in [
            "candidate_ms_positive",
            "w4a16_baseline_ms_positive",
            "best_reference_ms_positive",
            "speedup_vs_w4a16_ge_1p10",
            "speedup_vs_packed_p3_ge_1p00_or_unavailable",
            "median_speedup_vs_best_reference_ge_1p05",
            "no_shape_regression",
        ]
    )
    return {
        "lane": name,
        "shape": raw.get("shape", {}),
        "candidate_ms": candidate_ms,
        "baseline_w4a16_ms": w4a16_ms,
        "baseline_packed_p3_ms": p3_ms if _finite_positive(p3_ms) else None,
        "best_reference_ms": best_reference_ms,
        "speedup_vs_w4a16": speedup_vs_w4a16,
        "speedup_vs_packed_p3": speedup_vs_packed_p3 if _finite_positive(p3_ms) else None,
        "median_speedup_vs_best_reference": speedup_vs_best,
        "numeric_max_abs": numeric_max_abs,
        "numeric_rel_l2": numeric_rel_l2,
        "numeric_cosine": numeric_cosine,
        "passes": passes,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_r5c3c = Path(args.input_r5c3c).expanduser().resolve()
    candidate_json = Path(args.candidate_json).expanduser().resolve()
    r5c3c = _load_json(input_r5c3c)
    candidate = _load_json(candidate_json)
    raw_lanes = candidate.get("lanes") if isinstance(candidate.get("lanes"), dict) else {}
    lanes = {name: _lane_report(name, lane, args) for name, lane in sorted(raw_lanes.items()) if isinstance(lane, dict)}
    candidate_passes = candidate.get("passes") if isinstance(candidate.get("passes"), dict) else {}
    required_lanes = {"smoke", "production_shape"}
    lanes_present = set(lanes) >= required_lanes
    lane_numeric_ok = bool(lanes) and all((lane["passes"] or {}).get("numeric_parity_passed") is True for lane in lanes.values())
    lane_speed_ok = lanes_present and all((lane["passes"] or {}).get("speed_gate_passed") is True for lane in lanes.values())
    input_r5c3c_passed = r5c3c.get("decision") == "PASS_R5C3C_DOWN_WEIGHTED_PARITY_SMOKE"
    base_passes = {
        "input_r5c3c_passed": input_r5c3c_passed,
        "same_scope_ab": _as_bool(candidate_passes, "same_scope_ab"),
        "real_model_weights": _as_bool(candidate_passes, "real_model_weights"),
        "real_router_outputs": _as_bool(candidate_passes, "real_router_outputs"),
        "candidate_no_active_bf16_shadow": _as_bool(candidate_passes, "candidate_no_active_bf16_shadow"),
        "candidate_no_reload": _as_bool(candidate_passes, "candidate_no_reload"),
        "candidate_no_bf16_weight_materialization": _as_bool(candidate_passes, "candidate_no_bf16_weight_materialization"),
        "candidate_full_active_moe_boundary_timed": _as_bool(candidate_passes, "candidate_full_active_moe_boundary_timed"),
        "timing_includes_gateup_swiglu_down_weighted_scatter": _as_bool(
            candidate_passes, "timing_includes_gateup_swiglu_down_weighted_scatter"
        ),
        "numeric_vs_w4a16_or_p3_reference": lane_numeric_ok,
        "candidate_median_speedup_vs_best_reference_ge_1p05": lane_speed_ok,
        "required_lanes_present": lanes_present,
        "banked_full_active_moe_prefill_speed": False,
        "banked_grouped_moe_fp4_mma_poc": False,
        "banked_kernel_speed": False,
        "banked_decode_tps": False,
        "banked_server_rc": False,
        "banked_default_promotion": False,
        "banked_full_transformer_prefill": False,
    }
    non_speed_required = [
        "input_r5c3c_passed",
        "same_scope_ab",
        "real_model_weights",
        "real_router_outputs",
        "candidate_no_active_bf16_shadow",
        "candidate_no_reload",
        "candidate_no_bf16_weight_materialization",
        "candidate_full_active_moe_boundary_timed",
        "timing_includes_gateup_swiglu_down_weighted_scatter",
        "numeric_vs_w4a16_or_p3_reference",
        "required_lanes_present",
    ]
    non_speed_ok = all(bool(base_passes[key]) for key in non_speed_required)
    if non_speed_ok and lane_speed_ok:
        decision = PASS_DECISION
        base_passes["banked_full_active_moe_prefill_speed"] = True
        base_passes["banked_grouped_moe_fp4_mma_poc"] = True
        base_passes["banked_kernel_speed"] = True
    elif non_speed_ok:
        decision = DIAGNOSTIC_DECISION
    else:
        decision = FAIL_DECISION
    base_passes["all"] = decision == PASS_DECISION
    return {
        "schema": "lynn-stage6-r5c4-full-active-moe-prefill-speed-ab-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision": decision,
        "kernel_speed_scope": "active_moe_prefill_only",
        "input_r5c3c": str(input_r5c3c),
        "candidate_json": str(candidate_json),
        "candidate_schema": candidate.get("schema"),
        "lanes": lanes,
        "passes": base_passes,
        "promotion_boundary": {
            "full_active_moe_prefill_speed": bool(base_passes["banked_full_active_moe_prefill_speed"]),
            "grouped_moe_fp4_mma_poc": bool(base_passes["banked_grouped_moe_fp4_mma_poc"]),
            "kernel_speed": bool(base_passes["banked_kernel_speed"]),
            "decode_tps": False,
            "server_rc": False,
            "default_runtime": False,
            "full_transformer_prefill": False,
        },
        "thresholds": {
            "max_abs": args.max_abs_threshold,
            "rel_l2": args.rel_l2_threshold,
            "cosine": args.cosine_threshold,
            "speedup_vs_w4a16": 1.10,
            "speedup_vs_best_reference": 1.05,
        },
        "caveats": [
            "This validates a candidate-provided R5-C4 speed A/B artifact; it does not implement the kernel.",
            "A PASS is active-MoE prefill only, not Spark decode TPS, server/RC behavior, or default promotion.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-r5c3c", default=DEFAULT_R5C3C)
    ap.add_argument("--candidate-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-abs-threshold", type=float, default=1.0e-2)
    ap.add_argument("--rel-l2-threshold", type=float, default=1.0e-2)
    ap.add_argument("--cosine-threshold", type=float, default=0.999)
    args = ap.parse_args()
    data = run(args)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": data["decision"], "passes": data["passes"]}, indent=2))
    return 0 if data["decision"] in {PASS_DECISION, DIAGNOSTIC_DECISION} else 2


if __name__ == "__main__":
    raise SystemExit(main())
