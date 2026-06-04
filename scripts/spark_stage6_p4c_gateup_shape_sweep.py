#!/usr/bin/env python3
"""Stage 6 P4C gate/up launch-shape sweep.

This diagnostic gate sweeps the existing allocation-returning gate/up scalar
symbol over tile_inter/thread-count launch shapes. It checks whether a low-risk
parameter change is worth trying before replacing the gate/up half with a new
CUDA/CUTLASS speed candidate.
"""
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
    _bench_cuda,
    _byte_budget,
    _call_reference,
    _diff_stats,
    _env_snapshot,
    _make_inputs,
)

SCHEMA = "lynn-stage6-p4c-gateup-shape-sweep-v1"
PASS_DECISION = "PASS_P4C_GATEUP_SHAPE_SWEEP_RECORDED"
FAIL_DECISION = "FAIL_P4C_GATEUP_SHAPE_SWEEP"
BASELINE_SYMBOL = "gate_up_silu_tile_inter_scalar"
VARIANT_SYMBOL = "gate_up_silu_tile_inter_threads_scalar"


def _parse_int_csv(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _call_baseline(ext: Any, inputs: dict[str, torch.Tensor], tile_inter: int) -> torch.Tensor:
    return getattr(ext, BASELINE_SYMBOL)(
        inputs["hidden"].view(-1),
        inputs["expert_ids"].view(-1),
        inputs["gate_up_packed"],
        inputs["gate_up_scale"],
        inputs["gate_up_global_scale"],
        tile_inter,
    )


def _call_variant(
    ext: Any,
    inputs: dict[str, torch.Tensor],
    tile_inter: int,
    threads: int,
) -> torch.Tensor:
    return getattr(ext, VARIANT_SYMBOL)(
        inputs["hidden"].view(-1),
        inputs["expert_ids"].view(-1),
        inputs["gate_up_packed"],
        inputs["gate_up_scale"],
        inputs["gate_up_global_scale"],
        tile_inter,
        threads,
    )


def _diff_ok(diff: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        diff["finite_candidate"]
        and diff["finite_reference"]
        and diff["rel_l2"] <= args.rel_l2_threshold
        and diff["max_abs"] <= args.max_abs_threshold
    )


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "reference_symbol": REFERENCE_SYMBOL,
        "baseline_symbol": BASELINE_SYMBOL,
        "variant_symbol": VARIANT_SYMBOL,
        "tokens": args.tokens,
        "experts": args.experts,
        "top_k": args.top_k,
        "tile_tokens": args.tile_tokens,
        "baseline_tile_inter": args.baseline_tile_inter,
        "tile_inter_values": args.tile_inter_values,
        "thread_values": args.thread_values,
        "tile_hidden": args.tile_hidden,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "seed": args.seed,
        "actionable_speedup_floor": args.actionable_speedup_floor,
        "banked_p4c_gateup_shape_sweep": False,
        "banked_p4c_gateup_candidate": False,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        "promotion_policy": "diagnostic_launch_shape_sweep_only",
        "component_timing_caveat": "gate/up symbols allocate output tensors; use sweep for direction only",
        **_env_snapshot(),
    }
    if not torch.cuda.is_available():
        result["decision"] = "BLOCKED_NO_CUDA"
        result["passes"] = {"all": False, "cuda_available": False}
        return result

    build_dir = args.build_dir or os.environ.get(
        "LYNN_NATIVE_CUDA_BUILD_DIR",
        f"/tmp/lynn_engine_native_build/p4c_gateup_shape_sweep_{time.strftime('%Y%m%d_%H%M%S')}",
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

    for name in (REFERENCE_SYMBOL, BASELINE_SYMBOL, VARIANT_SYMBOL):
        result[f"{name}_present"] = bool(hasattr(ext, name))
    missing = [name for name in (REFERENCE_SYMBOL, BASELINE_SYMBOL, VARIANT_SYMBOL) if not hasattr(ext, name)]
    if missing:
        result["decision"] = "BLOCKED_SYMBOL_MISSING"
        result["missing_symbols"] = missing
        result["passes"] = {"all": False, "extension_loaded": True, "symbols_present": False}
        return result

    inputs = _make_inputs(args.tokens, args.experts, args.top_k, args.seed)
    result["byte_budget"] = _byte_budget({
        name: {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "bytes": int(t.numel() * t.element_size()),
            "contiguous": bool(t.is_contiguous()),
        }
        for name, t in inputs.items()
    }, experts=args.experts)

    _call_reference(ext, inputs, args)
    ref_inter = inputs["ref_inter_scratch"].view(args.top_k, 512)
    baseline_out = _call_baseline(ext, inputs, args.baseline_tile_inter)
    torch.cuda.synchronize()
    baseline_diff = _diff_stats(baseline_out, ref_inter)
    baseline_bench = _bench_cuda(
        lambda: _call_baseline(ext, inputs, args.baseline_tile_inter),
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
    )
    baseline_us = baseline_bench["median_us"]

    variants: list[dict[str, Any]] = []
    for tile_inter in args.tile_inter_values:
        for threads in args.thread_values:
            out = _call_variant(ext, inputs, tile_inter, threads)
            torch.cuda.synchronize()
            diff = _diff_stats(out, ref_inter)
            bench = _bench_cuda(
                lambda tile_inter=tile_inter, threads=threads: _call_variant(ext, inputs, tile_inter, threads),
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
            )
            median_us = bench["median_us"]
            numeric_ok = _diff_ok(diff, args)
            variants.append({
                "key": f"tile_inter_{tile_inter}_threads_{threads}",
                "tile_inter": tile_inter,
                "threads": threads,
                "median_us": median_us,
                "speedup_vs_current": baseline_us / median_us if median_us else None,
                "numeric_ok": bool(numeric_ok),
                "diff": diff,
                "bench": bench,
            })

    numeric_variants = [variant for variant in variants if variant.get("numeric_ok") and variant.get("median_us")]
    best = min(numeric_variants, key=lambda variant: float(variant["median_us"])) if numeric_variants else None
    best_speedup = best.get("speedup_vs_current") if best else None
    result["bench"] = {
        "current_baseline_gate_up": baseline_bench,
        "variants": variants,
        "best_variant": best,
        "best_speedup_vs_current": best_speedup,
        "best_is_actionable": bool(best_speedup is not None and best_speedup >= args.actionable_speedup_floor),
    }
    result["numeric_vs_reference"] = {
        "current_baseline_gate_up": baseline_diff,
        "variants_all_numeric_ok": bool(variants and all(variant.get("numeric_ok") for variant in variants)),
    }

    timing_ok = baseline_us > 0 and bool(variants) and all((variant.get("median_us") or 0) > 0 for variant in variants)
    numeric_ok = _diff_ok(baseline_diff, args) and result["numeric_vs_reference"]["variants_all_numeric_ok"]
    result["passes"] = {
        "all": bool(
            numeric_ok
            and timing_ok
            and result["banked_p4c_gateup_candidate"] is False
            and result["banked_fused_kernel"] is False
            and result["banked_default_promotion"] is False
        ),
        "extension_loaded": True,
        "symbols_present": True,
        "baseline_numeric_vs_reference": bool(_diff_ok(baseline_diff, args)),
        "variants_numeric_vs_reference": bool(result["numeric_vs_reference"]["variants_all_numeric_ok"]),
        "timing_recorded": bool(timing_ok),
        "promotion_boundary_closed": (
            result["banked_p4c_gateup_candidate"] is False
            and result["banked_fused_kernel"] is False
            and result["banked_default_promotion"] is False
        ),
    }
    result["banked_p4c_gateup_shape_sweep"] = bool(result["passes"]["all"])
    result["decision"] = PASS_DECISION if result["passes"]["all"] else FAIL_DECISION
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="P4C gate/up launch-shape sweep.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--tile-tokens", type=int, default=1)
    ap.add_argument("--baseline-tile-inter", type=int, default=8)
    ap.add_argument("--tile-inter-values", type=_parse_int_csv, default="1,2,4,8")
    ap.add_argument("--thread-values", type=_parse_int_csv, default="64,128,256")
    ap.add_argument("--tile-hidden", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--rel-l2-threshold", type=float, default=0.05)
    ap.add_argument("--max-abs-threshold", type=float, default=0.5)
    ap.add_argument("--actionable-speedup-floor", type=float, default=1.05)
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--strict-exit", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.tokens != 1 or args.top_k != 8:
        raise SystemExit("P4C gate/up shape sweep currently requires --tokens 1 --top-k 8")
    if args.experts < args.top_k:
        raise SystemExit("--experts must be >= --top-k")
    if args.baseline_tile_inter not in {1, 2, 4, 8}:
        raise SystemExit("--baseline-tile-inter must be one of {1, 2, 4, 8}")
    if any(value not in {1, 2, 4, 8} for value in args.tile_inter_values):
        raise SystemExit("--tile-inter-values may only contain {1, 2, 4, 8}")
    if any(value not in {64, 128, 256} for value in args.thread_values):
        raise SystemExit("--thread-values may only contain {64, 128, 256}")
    args.tile_inter = args.baseline_tile_inter

    started = time.time()
    result = run_sweep(args)
    result["elapsed_s"] = round(time.time() - started, 3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[p4c-gateup-shape-sweep] decision={result.get('decision')}")
    print(f"[p4c-gateup-shape-sweep] out={out}")
    if result.get("passes", {}).get("all"):
        return 0
    return 2 if args.strict_exit else 0


if __name__ == "__main__":
    raise SystemExit(main())
