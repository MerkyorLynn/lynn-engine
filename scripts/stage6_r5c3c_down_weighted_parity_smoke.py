#!/usr/bin/env python3
"""Stage 6 R5-C3C down projection + weighted top-k parity smoke.

R5-C3C consumes a banked R5-C3B gate/up value-materialization artifact. It
computes host SwiGLU, applies deterministic expert-specific down projections,
and performs route-weighted top-k reduction for both CUTLASS D values and the
CUTLASS host reference values.

This banks only numeric composition parity. It does not bank full grouped-MoE
FP4-MMA speed, decode TPS, server/RC behavior, or runtime defaults.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from r6000_stage6_r5c3b_gateup_value_materialization_smoke import (  # noqa: E402
    _fnv1a_f32,
    _parse_value_records,
    _pair_order,
    _scatter_values,
)
from r6000_stage6_r5c2_selected_expert_gateup_smoke import _build_routes  # noqa: E402


DEFAULT_INPUT = (
    "reports/stage6/"
    "r5c3b_gateup_value_materialization_smoke_20260604_204920/result.json"
)


def _load_input(result_path: Path) -> tuple[dict[str, Any], Path]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    value_file = Path(str(data.get("d_row_value_file") or ""))
    if not value_file.exists():
        sibling = result_path.with_name("result.d_row_values.jsonl")
        if sibling.exists():
            value_file = sibling
    return data, value_file


def _validate_r5c3b_input(data: dict[str, Any], value_file: Path) -> dict[str, bool]:
    passes = data.get("passes") if isinstance(data.get("passes"), dict) else {}
    return {
        "input_result_passed_r5c3b": data.get("decision") == "PASS_R5C3B_GATEUP_VALUE_MATERIALIZATION_SMOKE",
        "input_banked_gateup_value_materialization": passes.get("banked_gateup_value_materialization") is True,
        "input_banked_host_swiglu_checksum": passes.get("banked_host_swiglu_checksum_smoke") is True,
        "input_not_down_projection": passes.get("banked_down_projection_numeric_parity") is False,
        "input_not_speed": passes.get("banked_kernel_speed") is False,
        "input_value_file_exists": value_file.exists(),
    }


def _shape(data: dict[str, Any]) -> dict[str, Any]:
    shape = data.get("selected_expert_shape")
    if not isinstance(shape, dict):
        raise AssertionError("R5-C3B input missing selected_expert_shape")
    required = ["tokens", "top_k", "experts", "tokens_per_expert", "n_gate_up"]
    for key in required:
        if key not in shape:
            raise AssertionError(f"R5-C3B input shape missing {key}")
    return shape


def _swiglu(selected: list[list[list[float]]]) -> list[list[list[float]]]:
    output: list[list[list[float]]] = []
    for token in selected:
        token_out: list[list[float]] = []
        for row in token:
            if len(row) % 2:
                raise AssertionError("N_gateup must be even")
            half = len(row) // 2
            token_out.append([(gate / (1.0 + math.exp(-gate))) * up for gate, up in zip(row[:half], row[half:])])
        output.append(token_out)
    return output


def _down_weight(expert: int, hidden_idx: int, out_idx: int, hidden: int) -> float:
    # Integer-only deterministic pseudo weight. The scale keeps accumulation
    # finite while still making slot/weight perturbations visible.
    raw = ((expert + 1) * 1315423911 + (hidden_idx + 17) * 2654435761 + (out_idx + 31) * 97531) & 0xFFFF
    centered = (raw / 65535.0) - 0.5
    return centered / math.sqrt(max(hidden, 1))


def _expert_by_slot(pair_order: list[Any], tokens: int, top_k: int) -> list[list[int]]:
    experts = [[-1 for _slot in range(top_k)] for _token in range(tokens)]
    for pair in pair_order:
        experts[pair.token_idx][pair.top_k_slot] = pair.expert_id
    if any(expert < 0 for row in experts for expert in row):
        raise AssertionError("missing expert assignment for selected slot")
    return experts


def _down_project(
    swiglu: list[list[list[float]]],
    experts_by_slot: list[list[int]],
    out_dim: int,
) -> list[list[list[float]]]:
    output: list[list[list[float]]] = []
    for token_idx, token in enumerate(swiglu):
        token_out: list[list[float]] = []
        for slot_idx, row in enumerate(token):
            expert = experts_by_slot[token_idx][slot_idx]
            projected: list[float] = []
            for out_idx in range(out_dim):
                acc = 0.0
                for hidden_idx, value in enumerate(row):
                    acc += value * _down_weight(expert, hidden_idx, out_idx, len(row))
                projected.append(acc)
            token_out.append(projected)
        output.append(token_out)
    return output


def _route_weights(tokens: int, top_k: int) -> list[list[float]]:
    weights: list[list[float]] = []
    for token_idx in range(tokens):
        raw = [1.0 + float(((token_idx + 3) * (slot + 5) * 17) % 23) for slot in range(top_k)]
        total = sum(raw)
        weights.append([value / total for value in raw])
    return weights


def _weighted_topk(projected: list[list[list[float]]], weights: list[list[float]]) -> list[list[float]]:
    output: list[list[float]] = []
    for token_idx, token in enumerate(projected):
        out_dim = len(token[0])
        token_out = [0.0 for _ in range(out_dim)]
        for slot_idx, row in enumerate(token):
            weight = weights[token_idx][slot_idx]
            for out_idx, value in enumerate(row):
                token_out[out_idx] += weight * value
        output.append(token_out)
    return output


def _max_abs_nested(a: Any, b: Any) -> float:
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise AssertionError("nested length mismatch")
        return max((_max_abs_nested(x, y) for x, y in zip(a, b)), default=0.0)
    return abs(float(a) - float(b))


def _flatten(values: Any) -> list[float]:
    if isinstance(values, list):
        out: list[float] = []
        for item in values:
            out.extend(_flatten(item))
        return out
    return [float(values)]


def _parity_for_schedule(
    records: dict[tuple[int, int], dict[str, Any]],
    pair_order: list[Any],
    tokens: int,
    top_k: int,
    out_dim: int,
) -> dict[str, Any]:
    experts_by_slot = _expert_by_slot(pair_order, tokens, top_k)
    route_weights = _route_weights(tokens, top_k)
    d_selected = _scatter_values(records, pair_order, tokens, top_k, use_ref=False)
    ref_selected = _scatter_values(records, pair_order, tokens, top_k, use_ref=True)
    d_swiglu = _swiglu(d_selected)
    ref_swiglu = _swiglu(ref_selected)
    d_down = _down_project(d_swiglu, experts_by_slot, out_dim)
    ref_down = _down_project(ref_swiglu, experts_by_slot, out_dim)
    d_weighted = _weighted_topk(d_down, route_weights)
    ref_weighted = _weighted_topk(ref_down, route_weights)
    swiglu_max_abs = _max_abs_nested(d_swiglu, ref_swiglu)
    down_max_abs = _max_abs_nested(d_down, ref_down)
    weighted_max_abs = _max_abs_nested(d_weighted, ref_weighted)
    d_hash = _fnv1a_f32(_flatten(d_weighted))
    ref_hash = _fnv1a_f32(_flatten(ref_weighted))
    perturbed = json.loads(json.dumps(d_selected))
    perturbed[0][0][0] += 0.125
    perturbed_weighted = _weighted_topk(_down_project(_swiglu(perturbed), experts_by_slot, out_dim), route_weights)
    swapped_weights = [list(reversed(row)) for row in route_weights]
    swapped_weighted = _weighted_topk(d_down, swapped_weights)
    fault_checks = {
        "selected_value_perturbation_detected": _max_abs_nested(perturbed_weighted, d_weighted) > 0.0,
        "route_weight_swap_detected": _max_abs_nested(swapped_weighted, d_weighted) > 0.0,
    }
    return {
        "records": len(records),
        "out_dim": out_dim,
        "route_weight_sample": route_weights[:4],
        "swiglu_d_ref_max_abs": swiglu_max_abs,
        "down_d_ref_max_abs": down_max_abs,
        "weighted_topk_d_ref_max_abs": weighted_max_abs,
        "weighted_topk_d_hash": d_hash,
        "weighted_topk_ref_hash": ref_hash,
        "weighted_topk_checksum": sum(_flatten(d_weighted)),
        "fault_checks": fault_checks,
        "passes": {
            "swiglu_d_ref_match": swiglu_max_abs == 0.0,
            "down_projection_d_ref_match": down_max_abs == 0.0,
            "weighted_topk_d_ref_match": weighted_max_abs == 0.0,
            "weighted_topk_hash_match": d_hash == ref_hash,
            "fault_injections_detected": all(fault_checks.values()),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_result = Path(args.input_result).expanduser().resolve()
    input_data, value_file = _load_input(input_result)
    input_passes = _validate_r5c3b_input(input_data, value_file)
    shape = _shape(input_data)
    tokens = int(shape["tokens"])
    top_k = int(shape["top_k"])
    counts = [int(value) for value in shape["tokens_per_expert"]]
    n_gateup = int(shape["n_gate_up"])
    if n_gateup % 2:
        raise AssertionError("n_gate_up must be even")
    out_dim = int(args.out_dim)
    routes = _build_routes(tokens, top_k, counts)
    pair_order = _pair_order(routes, len(counts))
    records_by_schedule = _parse_value_records(value_file) if value_file.exists() else {}
    schedules = {
        name: _parity_for_schedule(records, pair_order, tokens, top_k, out_dim)
        for name, records in sorted(records_by_schedule.items())
    }
    schedule_passes = {name: all((schedule.get("passes") or {}).values()) for name, schedule in schedules.items()}
    checksums = [schedule.get("weighted_topk_checksum") for schedule in schedules.values()]
    passes = {
        **input_passes,
        "routes_reconstructed": len(routes) == tokens,
        "pair_order_complete": len(pair_order) == tokens * top_k,
        "schedules_available": set(schedules) == {"cooperative", "pingpong"},
        "schedule_parity_passed": bool(schedule_passes) and all(schedule_passes.values()),
        "schedule_weighted_checksums_match": len(set(checksums)) == 1 if checksums else False,
        "banked_down_projection_numeric_parity": False,
        "banked_weighted_topk_numeric_parity": False,
        "banked_grouped_moe_fp4_mma_poc": False,
        "banked_kernel_speed": False,
        "banked_default_promotion": False,
    }
    required = [
        "input_result_passed_r5c3b",
        "input_banked_gateup_value_materialization",
        "input_banked_host_swiglu_checksum",
        "input_not_down_projection",
        "input_not_speed",
        "input_value_file_exists",
        "routes_reconstructed",
        "pair_order_complete",
        "schedules_available",
        "schedule_parity_passed",
        "schedule_weighted_checksums_match",
    ]
    passes["banked_down_projection_numeric_parity"] = all(bool(passes[key]) for key in required)
    passes["banked_weighted_topk_numeric_parity"] = bool(passes["banked_down_projection_numeric_parity"])
    passes["all"] = bool(passes["banked_down_projection_numeric_parity"])
    decision = "PASS_R5C3C_DOWN_WEIGHTED_PARITY_SMOKE" if passes["all"] else "FAIL_R5C3C_DOWN_WEIGHTED_PARITY_SMOKE"
    return {
        "schema": "lynn-stage6-r5c3c-down-weighted-parity-smoke-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision": decision,
        "input_result": str(input_result),
        "input_value_file": str(value_file),
        "input_decision": input_data.get("decision"),
        "selected_expert_shape": {
            "tokens": tokens,
            "top_k": top_k,
            "experts": len(counts),
            "tokens_per_expert": counts,
            "n_gate_up": n_gateup,
            "swiglu_hidden": n_gateup // 2,
            "down_out_dim": out_dim,
        },
        "pair_order_sample": [asdict(pair) for pair in pair_order[: min(16, len(pair_order))]],
        "schedule_parity": schedules,
        "passes": passes,
        "promotion_boundary": {
            "down_projection_numeric_parity": bool(passes["banked_down_projection_numeric_parity"]),
            "weighted_topk_numeric_parity": bool(passes["banked_weighted_topk_numeric_parity"]),
            "grouped_moe_fp4_mma_poc": False,
            "kernel_speed": False,
            "decode_tps": False,
            "server_rc": False,
            "default_runtime": False,
        },
        "caveats": [
            "This is a host composition parity smoke using real R5-C3B CUTLASS gate/up D/ref values.",
            "Down weights and route weights are deterministic smoke weights, not model weights.",
            "This does not bank full active-MoE FP4-MMA speed, decode TPS, server/RC behavior, or defaults.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-result", default=DEFAULT_INPUT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-dim", type=int, default=48)
    args = ap.parse_args()
    data = run(args)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": data["decision"], "passes": data["passes"]}, indent=2))
    return 0 if data["passes"]["all"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
