#!/usr/bin/env python3
"""Stage 6 P4C active-reuse speed-baseline microbench.

P4C is the active-scratch-preserving replacement boundary after P4B's
single-kernel recompute attempts were rejected. This gate records the current
P4C symbol's speed baseline against the P4A two-stage reference on identical
synthetic packed-NVFP4 tensors. It does not bank fused speed or default
promotion.
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
    _make_inputs as _make_p4b_inputs,
    _tensor_manifest,
)

SCHEMA = "lynn-stage6-p4c-active-reuse-microbench-v1"
SYMBOL = "active_moe_fused_zero_shadow_active_reuse_contract"
PASS_DECISION = "PASS_P4C_ACTIVE_REUSE_SPEED_BASELINE_RECORDED"
FAIL_DECISION = "FAIL_P4C_ACTIVE_REUSE_SPEED_BASELINE"


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


def _make_inputs(tokens: int, experts: int, top_k: int, seed: int) -> dict[str, torch.Tensor]:
    inputs = _make_p4b_inputs(tokens, experts, top_k, seed)
    inputs["ref_inter_scratch"] = inputs.pop("inter_scratch").contiguous()
    inputs["candidate_inter_scratch"] = torch.empty_like(inputs["ref_inter_scratch"]).contiguous()
    return inputs


def _byte_budget(tensor_manifest: dict[str, Any], *, experts: int) -> dict[str, Any]:
    packed_weight_names = [
        "gate_up_packed",
        "gate_up_scale",
        "gate_up_global_scale",
        "down_packed",
        "down_scale",
        "down_global_scale",
    ]
    candidate_abi_names = [
        "hidden",
        "expert_ids",
        "routing_weights",
        "gate_up_packed",
        "gate_up_scale",
        "gate_up_global_scale",
        "down_packed",
        "down_scale",
        "down_global_scale",
        "candidate_inter_scratch",
        "candidate_out",
    ]
    forbidden_shadow_names = ["gate_up_proj", "down_proj", "gate_up_weight", "down_weight"]
    packed_weight_bytes = sum(int((tensor_manifest.get(name) or {}).get("bytes") or 0) for name in packed_weight_names)
    active_scratch_bytes = int((tensor_manifest.get("candidate_inter_scratch") or {}).get("bytes") or 0)
    bf16_shadow_equivalent_bytes = int(experts * ((1024 * 2048) + (2048 * 512)) * 2)
    forbidden_present = [
        name
        for name in candidate_abi_names
        if any(forbidden in name for forbidden in forbidden_shadow_names)
    ]
    return {
        "candidate_abi_names": candidate_abi_names,
        "packed_weight_names": packed_weight_names,
        "packed_weight_bytes": packed_weight_bytes,
        "active_scratch_bytes": active_scratch_bytes,
        "bf16_shadow_equivalent_bytes": bf16_shadow_equivalent_bytes,
        "packed_vs_bf16_shadow_ratio": (
            packed_weight_bytes / bf16_shadow_equivalent_bytes if bf16_shadow_equivalent_bytes else None
        ),
        "forbidden_shadow_tensor_names": forbidden_present,
        "zero_bf16_shadow_weight_abi": not forbidden_present,
        "active_scratch_reuse_abi": "candidate_inter_scratch" in candidate_abi_names and active_scratch_bytes > 0,
        "packed_byte_budget": packed_weight_bytes > 0 and packed_weight_bytes < bf16_shadow_equivalent_bytes,
    }


def _call_reference(ext: Any, inputs: dict[str, torch.Tensor], args: argparse.Namespace) -> None:
    getattr(ext, REFERENCE_SYMBOL)(
        inputs["hidden"],
        inputs["expert_ids"],
        inputs["routing_weights"],
        inputs["gate_up_packed"],
        inputs["gate_up_scale"],
        inputs["gate_up_global_scale"],
        inputs["down_packed"],
        inputs["down_scale"],
        inputs["down_global_scale"],
        inputs["ref_inter_scratch"],
        inputs["ref_out"],
        args.tile_tokens,
        args.tile_inter,
        args.tile_hidden,
    )


def _call_candidate(ext: Any, inputs: dict[str, torch.Tensor], args: argparse.Namespace) -> None:
    getattr(ext, SYMBOL)(
        inputs["hidden"],
        inputs["expert_ids"],
        inputs["routing_weights"],
        inputs["gate_up_packed"],
        inputs["gate_up_scale"],
        inputs["gate_up_global_scale"],
        inputs["down_packed"],
        inputs["down_scale"],
        inputs["down_global_scale"],
        inputs["candidate_inter_scratch"],
        inputs["candidate_out"],
        args.tile_tokens,
        args.tile_inter,
        args.tile_hidden,
    )


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


def _diff_stats(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    cand = candidate.detach().float()
    ref = reference.detach().float()
    diff = cand - ref
    ref_norm = ref.norm().clamp_min(1e-20)
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": float(diff.norm().item() / ref_norm.item()),
        "finite_candidate": bool(torch.isfinite(cand).all().item()),
        "finite_reference": bool(torch.isfinite(ref).all().item()),
    }


def run_microbench(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "symbol": SYMBOL,
        "reference_symbol": REFERENCE_SYMBOL,
        "tokens": args.tokens,
        "experts": args.experts,
        "top_k": args.top_k,
        "tile_tokens": args.tile_tokens,
        "tile_inter": args.tile_inter,
        "tile_hidden": args.tile_hidden,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "seed": args.seed,
        "speed_ratio_floor": args.speed_ratio_floor,
        "banked_p4c_active_reuse_speed_baseline": False,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        "promotion_policy": "measurement_only_active_reuse_baseline",
        **_env_snapshot(),
    }
    if not torch.cuda.is_available():
        result["decision"] = "BLOCKED_NO_CUDA"
        result["passes"] = {"all": False, "cuda_available": False}
        return result

    build_dir = args.build_dir or os.environ.get(
        "LYNN_NATIVE_CUDA_BUILD_DIR",
        f"/tmp/lynn_engine_native_build/p4c_active_reuse_microbench_{time.strftime('%Y%m%d_%H%M%S')}",
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
    out_diff = _diff_stats(inputs["candidate_out"], inputs["ref_out"])
    inter_diff = _diff_stats(inputs["candidate_inter_scratch"], inputs["ref_inter_scratch"])
    result["numeric_vs_reference"] = {
        "out": out_diff,
        "inter_scratch": inter_diff,
    }

    reference_bench = _bench_cuda(lambda: _call_reference(ext, inputs, args), warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    candidate_bench = _bench_cuda(lambda: _call_candidate(ext, inputs, args), warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    reference_us = reference_bench["median_us"]
    candidate_us = candidate_bench["median_us"]
    speedup = reference_us / candidate_us if candidate_us else None
    result["bench"] = {
        "reference_p4a_two_stage": reference_bench,
        "candidate_p4c_active_reuse_contract": candidate_bench,
        "candidate_vs_reference_speedup": speedup,
        "candidate_minus_reference_us": candidate_us - reference_us,
    }

    byte_budget = result["byte_budget"]
    numeric_ok = (
        out_diff["finite_candidate"]
        and out_diff["finite_reference"]
        and inter_diff["finite_candidate"]
        and inter_diff["finite_reference"]
        and out_diff["rel_l2"] <= args.rel_l2_threshold
        and out_diff["max_abs"] <= args.max_abs_threshold
        and inter_diff["rel_l2"] <= args.rel_l2_threshold
        and inter_diff["max_abs"] <= args.max_abs_threshold
    )
    measured_ok = reference_us > 0 and candidate_us > 0
    speed_floor_ok = speedup is not None and speedup >= args.speed_ratio_floor
    result["passes"] = {
        "all": bool(
            numeric_ok
            and measured_ok
            and speed_floor_ok
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
        "timing_recorded": bool(measured_ok),
        "speed_floor_recorded": bool(speed_floor_ok),
        "zero_bf16_shadow_weight_abi": byte_budget.get("zero_bf16_shadow_weight_abi") is True,
        "active_scratch_reuse_abi": byte_budget.get("active_scratch_reuse_abi") is True,
        "packed_byte_budget": byte_budget.get("packed_byte_budget") is True,
        "promotion_boundary_closed": result["banked_fused_kernel"] is False and result["banked_default_promotion"] is False,
    }
    result["banked_p4c_active_reuse_speed_baseline"] = bool(result["passes"]["all"])
    result["decision"] = PASS_DECISION if result["passes"]["all"] else FAIL_DECISION
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="P4C active-reuse speed-baseline microbench.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--tile-tokens", type=int, default=1)
    ap.add_argument("--tile-inter", type=int, default=8)
    ap.add_argument("--tile-hidden", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--rel-l2-threshold", type=float, default=0.05)
    ap.add_argument("--max-abs-threshold", type=float, default=0.5)
    ap.add_argument("--speed-ratio-floor", type=float, default=0.8)
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--strict-exit", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.tokens != 1 or args.top_k != 8:
        raise SystemExit("P4C active-reuse microbench currently requires --tokens 1 --top-k 8")
    if args.experts < args.top_k:
        raise SystemExit("--experts must be >= --top-k")

    started = time.time()
    result = run_microbench(args)
    result["elapsed_s"] = round(time.time() - started, 3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[p4c-active-reuse-microbench] decision={result.get('decision')}")
    print(f"[p4c-active-reuse-microbench] out={out}")
    if result.get("passes", {}).get("all"):
        return 0
    return 2 if args.strict_exit else 0


if __name__ == "__main__":
    raise SystemExit(main())
