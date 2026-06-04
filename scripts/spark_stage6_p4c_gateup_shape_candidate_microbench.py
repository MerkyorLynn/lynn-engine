#!/usr/bin/env python3
"""Stage 6 P4C gate/up launch-shape speed-candidate microbench."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.spark_stage6_p4c_active_reuse_microbench import (  # noqa: E402
    REFERENCE_SYMBOL,
    SCHEMA as BASELINE_SCHEMA,
    SYMBOL,
    _bench_cuda,
    _byte_budget,
    _call_candidate,
    _call_reference,
    _diff_stats,
    _env_snapshot,
    _make_inputs,
)
from scripts.spark_stage6_p4b_single_cta_numeric_preflight import _tensor_manifest  # noqa: E402

SCHEMA = "lynn-stage6-p4c-gateup-shape-candidate-microbench-v1"
PASS_DECISION = "PASS_P4C_GATEUP_SHAPE_CANDIDATE_RECORDED"
FAIL_DECISION = "FAIL_P4C_GATEUP_SHAPE_CANDIDATE"


def _with_tile_inter(args: argparse.Namespace, tile_inter: int) -> argparse.Namespace:
    data = vars(args).copy()
    data["tile_inter"] = tile_inter
    return argparse.Namespace(**data)


def _diff_ok(diff: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        diff["finite_candidate"]
        and diff["finite_reference"]
        and diff["rel_l2"] <= args.rel_l2_threshold
        and diff["max_abs"] <= args.max_abs_threshold
    )


def run_microbench(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "baseline_schema_reference": BASELINE_SCHEMA,
        "symbol": SYMBOL,
        "reference_symbol": REFERENCE_SYMBOL,
        "tokens": args.tokens,
        "experts": args.experts,
        "top_k": args.top_k,
        "tile_tokens": args.tile_tokens,
        "current_tile_inter": args.current_tile_inter,
        "candidate_tile_inter": args.candidate_tile_inter,
        "tile_hidden": args.tile_hidden,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "seed": args.seed,
        "candidate_speedup_floor": args.candidate_speedup_floor,
        "banked_p4c_gateup_shape_candidate": False,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        "promotion_policy": "opt_in_candidate_microbench_only",
        **_env_snapshot(),
    }
    if not torch.cuda.is_available():
        result["decision"] = "BLOCKED_NO_CUDA"
        result["passes"] = {"all": False, "cuda_available": False}
        return result

    build_dir = args.build_dir or os.environ.get(
        "LYNN_NATIVE_CUDA_BUILD_DIR",
        f"/tmp/lynn_engine_native_build/p4c_gateup_shape_candidate_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    result["build_dir"] = build_dir

    try:
        from engine.native_cuda import load_lynn_native_extension

        ext = load_lynn_native_extension(build_dir=build_dir, verbose=args.verbose)
        result["extension_loaded"] = True
    except Exception as exc:
        result["extension_loaded"] = False
        result["load_error_tail"] = str(exc)[-2000:]
        result["decision"] = "BLOCKED_COMPILE"
        result["passes"] = {"all": False, "extension_loaded": False}
        return result

    result["symbol_present"] = bool(hasattr(ext, SYMBOL))
    result["reference_symbol_present"] = bool(hasattr(ext, REFERENCE_SYMBOL))
    if not result["symbol_present"] or not result["reference_symbol_present"]:
        result["decision"] = "BLOCKED_SYMBOL_MISSING"
        result["passes"] = {
            "all": False,
            "extension_loaded": True,
            "symbol_present": result["symbol_present"],
            "reference_symbol_present": result["reference_symbol_present"],
        }
        return result

    inputs = _make_inputs(args.tokens, args.experts, args.top_k, args.seed)
    result["tensor_manifest"] = _tensor_manifest(inputs)
    result["byte_budget"] = _byte_budget(result["tensor_manifest"], experts=args.experts)
    current_args = _with_tile_inter(args, args.current_tile_inter)
    candidate_args = _with_tile_inter(args, args.candidate_tile_inter)

    _call_reference(ext, inputs, candidate_args)
    _call_candidate(ext, inputs, candidate_args)
    torch.cuda.synchronize()
    candidate_out_diff = _diff_stats(inputs["candidate_out"], inputs["ref_out"])
    candidate_inter_diff = _diff_stats(inputs["candidate_inter_scratch"], inputs["ref_inter_scratch"])

    _call_reference(ext, inputs, current_args)
    current_ref_out = inputs["ref_out"].clone()
    current_ref_inter = inputs["ref_inter_scratch"].clone()
    _call_reference(ext, inputs, candidate_args)
    reference_tile_change_out_diff = _diff_stats(inputs["ref_out"], current_ref_out)
    reference_tile_change_inter_diff = _diff_stats(inputs["ref_inter_scratch"], current_ref_inter)
    result["numeric_vs_reference"] = {
        "candidate_vs_p4a_candidate_tile_out": candidate_out_diff,
        "candidate_vs_p4a_candidate_tile_inter_scratch": candidate_inter_diff,
        "p4a_candidate_tile_vs_current_tile_out": reference_tile_change_out_diff,
        "p4a_candidate_tile_vs_current_tile_inter_scratch": reference_tile_change_inter_diff,
    }

    reference_current_bench = _bench_cuda(
        lambda: _call_reference(ext, inputs, current_args),
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
    )
    reference_candidate_bench = _bench_cuda(
        lambda: _call_reference(ext, inputs, candidate_args),
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
    )
    current_bench = _bench_cuda(
        lambda: _call_candidate(ext, inputs, current_args),
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
    )
    candidate_bench = _bench_cuda(
        lambda: _call_candidate(ext, inputs, candidate_args),
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
    )
    current_us = current_bench["median_us"]
    candidate_us = candidate_bench["median_us"]
    result["bench"] = {
        "reference_p4a_current_tile": reference_current_bench,
        "reference_p4a_candidate_tile": reference_candidate_bench,
        "current_p4c_active_reuse_contract": current_bench,
        "candidate_p4c_active_reuse_contract": candidate_bench,
        "candidate_vs_current_speedup": current_us / candidate_us if candidate_us else None,
        "candidate_minus_current_us": candidate_us - current_us,
        "reference_candidate_vs_current_speedup": (
            reference_current_bench["median_us"] / reference_candidate_bench["median_us"]
            if reference_candidate_bench["median_us"]
            else None
        ),
    }

    byte_budget = result["byte_budget"]
    numeric_ok = _diff_ok(candidate_out_diff, args) and _diff_ok(candidate_inter_diff, args)
    timing_ok = current_us > 0 and candidate_us > 0
    speedup = result["bench"]["candidate_vs_current_speedup"]
    speed_ok = speedup is not None and speedup >= args.candidate_speedup_floor
    result["passes"] = {
        "all": bool(
            numeric_ok
            and timing_ok
            and speed_ok
            and byte_budget.get("zero_bf16_shadow_weight_abi") is True
            and byte_budget.get("active_scratch_reuse_abi") is True
            and byte_budget.get("packed_byte_budget") is True
            and result["banked_fused_kernel"] is False
            and result["banked_default_promotion"] is False
        ),
        "extension_loaded": True,
        "symbol_present": result["symbol_present"],
        "reference_symbol_present": result["reference_symbol_present"],
        "numeric_vs_reference": bool(numeric_ok),
        "timing_recorded": bool(timing_ok),
        "candidate_speed_floor": bool(speed_ok),
        "zero_bf16_shadow_weight_abi": byte_budget.get("zero_bf16_shadow_weight_abi") is True,
        "active_scratch_reuse_abi": byte_budget.get("active_scratch_reuse_abi") is True,
        "packed_byte_budget": byte_budget.get("packed_byte_budget") is True,
        "promotion_boundary_closed": result["banked_fused_kernel"] is False and result["banked_default_promotion"] is False,
    }
    result["banked_p4c_gateup_shape_candidate"] = bool(result["passes"]["all"])
    result["decision"] = PASS_DECISION if result["passes"]["all"] else FAIL_DECISION
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="P4C gate/up launch-shape speed-candidate microbench.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--tile-tokens", type=int, default=1)
    ap.add_argument("--current-tile-inter", type=int, default=8)
    ap.add_argument("--candidate-tile-inter", type=int, default=2)
    ap.add_argument("--tile-hidden", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--rel-l2-threshold", type=float, default=0.05)
    ap.add_argument("--max-abs-threshold", type=float, default=0.5)
    ap.add_argument("--candidate-speedup-floor", type=float, default=1.05)
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--strict-exit", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.tokens != 1 or args.top_k != 8:
        raise SystemExit("P4C gate/up shape candidate currently requires --tokens 1 --top-k 8")
    if args.experts < args.top_k:
        raise SystemExit("--experts must be >= --top-k")
    if args.current_tile_inter not in {1, 2, 4, 8} or args.candidate_tile_inter not in {1, 2, 4, 8}:
        raise SystemExit("--current-tile-inter/--candidate-tile-inter must be in {1, 2, 4, 8}")
    if args.tile_hidden not in {1, 2, 4, 8}:
        raise SystemExit("--tile-hidden must be in {1, 2, 4, 8}")

    started = time.time()
    result = run_microbench(args)
    result["elapsed_s"] = round(time.time() - started, 3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[p4c-gateup-shape-candidate] decision={result.get('decision')}")
    print(f"[p4c-gateup-shape-candidate] out={out}")
    if result.get("passes", {}).get("all"):
        return 0
    return 2 if args.strict_exit else 0


if __name__ == "__main__":
    raise SystemExit(main())
