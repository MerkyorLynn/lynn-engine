#!/usr/bin/env python3
"""Stage 6 P4C active-reuse component profile.

This gate decomposes the banked P4C active-reuse baseline into the existing
gate/up and down component symbols. It is diagnostic only: component symbols
return newly allocated tensors, so their timings are not promotion evidence.
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
    SCHEMA as BASELINE_SCHEMA,
    SYMBOL,
    _bench_cuda,
    _byte_budget,
    _call_reference,
    _diff_stats,
    _env_snapshot,
    _make_inputs,
)

SCHEMA = "lynn-stage6-p4c-component-profile-v1"
PASS_DECISION = "PASS_P4C_COMPONENT_PROFILE_RECORDED"
FAIL_DECISION = "FAIL_P4C_COMPONENT_PROFILE"
GATE_SYMBOL = "gate_up_silu_tile_inter_scalar"
DOWN_SYMBOL = "down_weighted_sum_tile_scalar"


def _call_gate(ext: Any, inputs: dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    return getattr(ext, GATE_SYMBOL)(
        inputs["hidden"].view(-1),
        inputs["expert_ids"].view(-1),
        inputs["gate_up_packed"],
        inputs["gate_up_scale"],
        inputs["gate_up_global_scale"],
        args.tile_inter,
    )


def _call_down(
    ext: Any,
    inter: torch.Tensor,
    inputs: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> torch.Tensor:
    return getattr(ext, DOWN_SYMBOL)(
        inter,
        inputs["expert_ids"].view(-1),
        inputs["routing_weights"].view(-1),
        inputs["down_packed"],
        inputs["down_scale"],
        inputs["down_global_scale"],
        args.tile_hidden,
    ).view(1, -1)


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "baseline_schema_reference": BASELINE_SCHEMA,
        "symbol": SYMBOL,
        "reference_symbol": REFERENCE_SYMBOL,
        "gate_symbol": GATE_SYMBOL,
        "down_symbol": DOWN_SYMBOL,
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
        "banked_p4c_component_profile": False,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        "promotion_policy": "diagnostic_component_profile_only",
        "component_timing_caveat": "component symbols allocate output tensors; use split for bottleneck direction only",
        **_env_snapshot(),
    }
    if not torch.cuda.is_available():
        result["decision"] = "BLOCKED_NO_CUDA"
        result["passes"] = {"all": False, "cuda_available": False}
        return result

    build_dir = args.build_dir or os.environ.get(
        "LYNN_NATIVE_CUDA_BUILD_DIR",
        f"/tmp/lynn_engine_native_build/p4c_component_profile_{time.strftime('%Y%m%d_%H%M%S')}",
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

    for name in (SYMBOL, REFERENCE_SYMBOL, GATE_SYMBOL, DOWN_SYMBOL):
        result[f"{name}_present"] = bool(hasattr(ext, name))
    missing = [name for name in (SYMBOL, REFERENCE_SYMBOL, GATE_SYMBOL, DOWN_SYMBOL) if not hasattr(ext, name)]
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
    gate_out = _call_gate(ext, inputs, args)
    down_out = _call_down(ext, inputs["ref_inter_scratch"].view(args.top_k, 512), inputs, args)
    down_from_gate_out = _call_down(ext, gate_out, inputs, args)
    torch.cuda.synchronize()

    gate_diff = _diff_stats(gate_out, inputs["ref_inter_scratch"].view(args.top_k, 512))
    down_diff = _diff_stats(down_out, inputs["ref_out"])
    composed_diff = _diff_stats(down_from_gate_out, inputs["ref_out"])
    result["numeric_vs_reference"] = {
        "gate_inter_scratch": gate_diff,
        "down_on_ref_scratch": down_diff,
        "gate_plus_down_composed": composed_diff,
    }

    full_bench = _bench_cuda(lambda: _call_reference(ext, inputs, args), warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    gate_bench = _bench_cuda(lambda: _call_gate(ext, inputs, args), warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    down_bench = _bench_cuda(
        lambda: _call_down(ext, inputs["ref_inter_scratch"].view(args.top_k, 512), inputs, args),
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
    )
    full_us = full_bench["median_us"]
    gate_us = gate_bench["median_us"]
    down_us = down_bench["median_us"]
    component_sum_us = gate_us + down_us
    result["bench"] = {
        "full_p4c_active_reuse_contract": full_bench,
        "component_gate_up_allocating": gate_bench,
        "component_down_allocating": down_bench,
        "component_sum_us": component_sum_us,
        "gate_share_of_component_sum": gate_us / component_sum_us if component_sum_us else None,
        "down_share_of_component_sum": down_us / component_sum_us if component_sum_us else None,
        "component_sum_vs_full_ratio": component_sum_us / full_us if full_us else None,
    }

    def diff_ok(diff: dict[str, Any]) -> bool:
        return (
            diff["finite_candidate"]
            and diff["finite_reference"]
            and diff["rel_l2"] <= args.rel_l2_threshold
            and diff["max_abs"] <= args.max_abs_threshold
        )

    numeric_ok = diff_ok(gate_diff) and diff_ok(down_diff) and diff_ok(composed_diff)
    timing_ok = full_us > 0 and gate_us > 0 and down_us > 0
    result["passes"] = {
        "all": bool(
            numeric_ok
            and timing_ok
            and result["banked_fused_kernel"] is False
            and result["banked_default_promotion"] is False
        ),
        "extension_loaded": True,
        "symbols_present": True,
        "numeric_vs_reference": bool(numeric_ok),
        "timing_recorded": bool(timing_ok),
        "promotion_boundary_closed": result["banked_fused_kernel"] is False and result["banked_default_promotion"] is False,
    }
    result["banked_p4c_component_profile"] = bool(result["passes"]["all"])
    result["decision"] = PASS_DECISION if result["passes"]["all"] else FAIL_DECISION
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="P4C active-reuse component profile.")
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
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--strict-exit", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.tokens != 1 or args.top_k != 8:
        raise SystemExit("P4C component profile currently requires --tokens 1 --top-k 8")
    if args.experts < args.top_k:
        raise SystemExit("--experts must be >= --top-k")

    started = time.time()
    result = run_profile(args)
    result["elapsed_s"] = round(time.time() - started, 3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[p4c-component-profile] decision={result.get('decision')}")
    print(f"[p4c-component-profile] out={out}")
    if result.get("passes", {}).get("all"):
        return 0
    return 2 if args.strict_exit else 0


if __name__ == "__main__":
    raise SystemExit(main())
