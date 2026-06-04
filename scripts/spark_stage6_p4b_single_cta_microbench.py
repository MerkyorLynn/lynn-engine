#!/usr/bin/env python3
"""Stage 6 P4B single-CTA microbench.

This measures the opt-in P4B single-CTA numeric reference against the P4A
two-stage caller-owned reference on identical synthetic packed-NVFP4 tensors.
It records timing evidence only; it must keep fused-kernel/default promotion
closed even if the tiny fixture looks favorable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.spark_stage6_p4b_single_cta_numeric_preflight import (  # noqa: E402
    REFERENCE_SYMBOL,
    SCHEMA as NUMERIC_SCHEMA,
    SYMBOL,
    _byte_budget,
    _call_candidate,
    _call_reference,
    _make_inputs,
    _tensor_manifest,
)

SCHEMA = "lynn-stage6-p4b-single-cta-microbench-v1"


def _env_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "lynn_native_cuda_arch": os.environ.get("LYNN_NATIVE_CUDA_ARCH"),
        "lynn_native_cuda_arch_auto": os.environ.get("LYNN_NATIVE_CUDA_ARCH_AUTO"),
        "lynn_native_cuda_build_dir": os.environ.get("LYNN_NATIVE_CUDA_BUILD_DIR"),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
    return info


def _bench_cuda(fn: Callable[[], None], *, warmup: int, iters: int, repeats: int) -> dict[str, Any]:
    times_us: list[float] = []
    for _ in range(repeats):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times_us.append(float(start.elapsed_time(end) * 1000.0 / iters))
    return {
        "warmup": warmup,
        "iters": iters,
        "repeats": repeats,
        "median_us": float(statistics.median(times_us)),
        "min_us": float(min(times_us)),
        "max_us": float(max(times_us)),
        "all_us": times_us,
    }


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    af = a.detach().float()
    bf = b.detach().float()
    diff = af - bf
    ref_norm = bf.norm().clamp_min(1e-20)
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": float(diff.norm().item() / ref_norm.item()),
        "finite_a": bool(torch.isfinite(af).all().item()),
        "finite_b": bool(torch.isfinite(bf).all().item()),
    }


def run_microbench(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "numeric_schema_reference": NUMERIC_SCHEMA,
        "symbol": SYMBOL,
        "reference_symbol": REFERENCE_SYMBOL,
        "tokens": args.tokens,
        "experts": args.experts,
        "top_k": args.top_k,
        "candidate_mode": args.candidate_mode,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "seed": args.seed,
        "banked_single_cta_microbench": False,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        "promotion_policy": "measurement_only_reference_path",
        **_env_snapshot(),
    }
    if not torch.cuda.is_available():
        result["decision"] = "BLOCKED_NO_CUDA"
        result["passes"] = {"all": False, "cuda_available": False}
        return result

    build_dir = args.build_dir or os.environ.get(
        "LYNN_NATIVE_CUDA_BUILD_DIR",
        f"/tmp/lynn_engine_native_build/p4b_single_cta_microbench_{time.strftime('%Y%m%d_%H%M%S')}",
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

    _call_reference(ext, inputs, args)
    _call_candidate(ext, inputs, args)
    torch.cuda.synchronize()
    diff = _diff_stats(inputs["candidate_out"], inputs["ref_out"])
    result["numeric_vs_reference"] = diff

    reference_bench = _bench_cuda(lambda: _call_reference(ext, inputs, args), warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    candidate_bench = _bench_cuda(lambda: _call_candidate(ext, inputs, args), warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    reference_us = reference_bench["median_us"]
    candidate_us = candidate_bench["median_us"]
    result["bench"] = {
        "reference_p4a_two_stage": reference_bench,
        "candidate_p4b_single_cta": candidate_bench,
        "candidate_vs_reference_speedup": (reference_us / candidate_us if candidate_us else None),
        "candidate_minus_reference_us": candidate_us - reference_us,
    }

    byte_budget = result["byte_budget"]
    numeric_ok = (
        diff["finite_a"]
        and diff["finite_b"]
        and diff["rel_l2"] <= args.rel_l2_threshold
        and diff["max_abs"] <= args.max_abs_threshold
    )
    measured_ok = reference_us > 0 and candidate_us > 0
    result["passes"] = {
        "all": bool(numeric_ok and measured_ok and byte_budget.get("no_inter_scratch_candidate_abi") is True),
        "extension_loaded": True,
        "symbol_present": result["symbol_present"],
        "reference_symbol_present": result["reference_symbol_present"],
        "numeric_vs_reference": bool(numeric_ok),
        "timing_recorded": bool(measured_ok),
        "no_inter_scratch_candidate_abi": byte_budget.get("no_inter_scratch_candidate_abi") is True,
        "promotion_boundary_closed": result["banked_fused_kernel"] is False and result["banked_default_promotion"] is False,
    }
    result["banked_single_cta_microbench"] = bool(result["passes"]["all"])
    result["decision"] = (
        "PASS_P4B_SINGLE_CTA_MICROBENCH_RECORDED"
        if result["passes"]["all"]
        else "FAIL_P4B_SINGLE_CTA_MICROBENCH"
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="P4B single-CTA microbench.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--tile-tokens", type=int, default=1)
    ap.add_argument("--tile-inter", type=int, default=8)
    ap.add_argument("--tile-experts", type=int, default=1)
    ap.add_argument("--tile-hidden", type=int, default=8)
    ap.add_argument("--candidate-mode", choices=["single_cta", "multi_cta"], default="single_cta")
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--rel-l2-threshold", type=float, default=0.05)
    ap.add_argument("--max-abs-threshold", type=float, default=0.5)
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--strict-exit", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.tokens != 1 or args.top_k != 8:
        raise SystemExit("P4B single-CTA microbench currently requires --tokens 1 --top-k 8")
    if args.experts < args.top_k:
        raise SystemExit("--experts must be >= --top-k")

    started = time.time()
    result = run_microbench(args)
    result["elapsed_s"] = round(time.time() - started, 3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[p4b-single-cta-microbench] decision={result.get('decision')}")
    print(f"[p4b-single-cta-microbench] out={out}")
    if result.get("passes", {}).get("all"):
        return 0
    return 2 if args.strict_exit else 0


if __name__ == "__main__":
    raise SystemExit(main())
