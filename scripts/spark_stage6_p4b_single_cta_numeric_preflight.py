#!/usr/bin/env python3
"""Stage 6 P4B single-CTA numeric preflight.

This gate opts into the first output-returning P4B implementation via
LYNN_NATIVE_P4B_SINGLE_CTA_REFERENCE=1, then compares it against the P4A
caller-owned two-stage reference on the same synthetic packed-NVFP4 tensors.
It banks implementation correctness only; speed/default promotion stay closed.
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

SCHEMA = "lynn-stage6-p4b-single-cta-numeric-preflight-v1"
SYMBOL = "active_moe_fused_zero_shadow_single_kernel_contract"
REFERENCE_SYMBOL = "active_moe_fused_zero_shadow_out_contract"


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
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    hidden = (torch.randn(tokens, 2048, dtype=torch.float32, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    expert_ids = torch.arange(top_k, dtype=torch.int32, device="cuda").reshape(1, top_k).repeat(tokens, 1)
    routing_weights = torch.rand(tokens, top_k, dtype=torch.float32, device="cuda", generator=gen)
    routing_weights = routing_weights / routing_weights.sum(dim=1, keepdim=True)
    gate_up_packed = torch.randint(0, 16, (experts, 1024, 1024), dtype=torch.uint8, device="cuda", generator=gen)
    down_packed = torch.randint(0, 16, (experts, 2048, 256), dtype=torch.uint8, device="cuda", generator=gen)
    # Keep scales small so the synthetic fixture stays finite but non-trivial.
    gate_up_scale = torch.full((experts, 1024, 128), 0.003, dtype=torch.float32, device="cuda")
    down_scale = torch.full((experts, 2048, 32), 0.003, dtype=torch.float32, device="cuda")
    gate_up_global_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    down_global_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    inter_scratch = torch.empty(tokens, top_k, 512, dtype=torch.bfloat16, device="cuda")
    ref_out = torch.empty_like(hidden)
    candidate_out = torch.empty_like(hidden)
    return {
        "hidden": hidden.contiguous(),
        "expert_ids": expert_ids.contiguous(),
        "routing_weights": routing_weights.contiguous(),
        "gate_up_packed": gate_up_packed.contiguous(),
        "gate_up_scale": gate_up_scale.contiguous(),
        "gate_up_global_scale": gate_up_global_scale.contiguous(),
        "down_packed": down_packed.contiguous(),
        "down_scale": down_scale.contiguous(),
        "down_global_scale": down_global_scale.contiguous(),
        "inter_scratch": inter_scratch.contiguous(),
        "ref_out": ref_out.contiguous(),
        "candidate_out": candidate_out.contiguous(),
    }


def _tensor_manifest(inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "bytes": int(t.numel() * t.element_size()),
            "contiguous": bool(t.is_contiguous()),
        }
        for name, t in inputs.items()
    }


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
        "candidate_out",
    ]
    forbidden_candidate_names = ["inter_scratch", "ref_out", "gate_up_proj", "down_proj", "gate_up_weight", "down_weight"]
    packed_weight_bytes = sum(int((tensor_manifest.get(name) or {}).get("bytes") or 0) for name in packed_weight_names)
    bf16_shadow_equivalent_bytes = int(experts * ((1024 * 2048) + (2048 * 512)) * 2)
    forbidden_present = [
        name
        for name in candidate_abi_names
        if any(forbidden in name for forbidden in forbidden_candidate_names)
    ]
    return {
        "candidate_abi_names": candidate_abi_names,
        "packed_weight_names": packed_weight_names,
        "packed_weight_bytes": packed_weight_bytes,
        "bf16_shadow_equivalent_bytes": bf16_shadow_equivalent_bytes,
        "packed_vs_bf16_shadow_ratio": (
            packed_weight_bytes / bf16_shadow_equivalent_bytes if bf16_shadow_equivalent_bytes else None
        ),
        "forbidden_candidate_tensor_names": forbidden_present,
        "zero_shadow_candidate_abi": not forbidden_present,
        "packed_byte_budget": packed_weight_bytes > 0 and packed_weight_bytes < bf16_shadow_equivalent_bytes,
        "no_inter_scratch_candidate_abi": "inter_scratch" not in candidate_abi_names,
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
        inputs["inter_scratch"],
        inputs["ref_out"],
        args.tile_tokens,
        args.tile_inter,
        args.tile_hidden,
    )


def _call_candidate(ext: Any, inputs: dict[str, torch.Tensor], args: argparse.Namespace) -> None:
    old = os.environ.get("LYNN_NATIVE_P4B_SINGLE_CTA_REFERENCE")
    os.environ["LYNN_NATIVE_P4B_SINGLE_CTA_REFERENCE"] = "1"
    try:
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
            inputs["candidate_out"],
            args.tile_tokens,
            args.tile_experts,
            args.tile_hidden,
        )
    finally:
        if old is None:
            os.environ.pop("LYNN_NATIVE_P4B_SINGLE_CTA_REFERENCE", None)
        else:
            os.environ["LYNN_NATIVE_P4B_SINGLE_CTA_REFERENCE"] = old


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "symbol": SYMBOL,
        "reference_symbol": REFERENCE_SYMBOL,
        "tokens": args.tokens,
        "experts": args.experts,
        "top_k": args.top_k,
        "tile_tokens": args.tile_tokens,
        "tile_inter": args.tile_inter,
        "tile_experts": args.tile_experts,
        "tile_hidden": args.tile_hidden,
        "seed": args.seed,
        "banked_single_cta_numeric_preflight": False,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        **_env_snapshot(),
    }
    if not torch.cuda.is_available():
        result["decision"] = "BLOCKED_NO_CUDA"
        result["passes"] = {"all": False, "cuda_available": False}
        return result

    build_dir = args.build_dir or os.environ.get(
        "LYNN_NATIVE_CUDA_BUILD_DIR",
        f"/tmp/lynn_engine_native_build/p4b_single_cta_numeric_{time.strftime('%Y%m%d_%H%M%S')}",
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

    reference_error = None
    candidate_error = None
    try:
        _call_reference(ext, inputs, args)
        torch.cuda.synchronize()
    except Exception as exc:
        reference_error = {"type": type(exc).__name__, "message": str(exc)[-2000:]}
    try:
        _call_candidate(ext, inputs, args)
        torch.cuda.synchronize()
    except Exception as exc:
        candidate_error = {"type": type(exc).__name__, "message": str(exc)[-2000:]}
    result["reference_error"] = reference_error
    result["candidate_error"] = candidate_error

    ref = inputs["ref_out"].float()
    cand = inputs["candidate_out"].float()
    diff = (cand - ref).abs()
    ref_norm = ref.norm().clamp_min(1e-20)
    candidate_finite = bool(torch.isfinite(cand).all().item()) if candidate_error is None else False
    reference_finite = bool(torch.isfinite(ref).all().item()) if reference_error is None else False
    max_abs = float(diff.max().item()) if reference_error is None and candidate_error is None else None
    mean_abs = float(diff.mean().item()) if reference_error is None and candidate_error is None else None
    rel_l2 = float((cand - ref).norm().item() / ref_norm.item()) if reference_error is None and candidate_error is None else None
    result["reference_output"] = {
        "shape": list(inputs["ref_out"].shape),
        "dtype": str(inputs["ref_out"].dtype),
        "finite": reference_finite,
        "norm": float(ref.norm().item()) if reference_error is None else None,
    }
    result["candidate_output"] = {
        "shape": list(inputs["candidate_out"].shape),
        "dtype": str(inputs["candidate_out"].dtype),
        "finite": candidate_finite,
        "norm": float(cand.norm().item()) if candidate_error is None else None,
        "max_abs_diff_vs_reference": max_abs,
        "mean_abs_diff_vs_reference": mean_abs,
        "rel_l2_vs_reference": rel_l2,
    }
    zero_shadow_candidate_abi = bool(result["byte_budget"]["zero_shadow_candidate_abi"])
    packed_byte_budget = bool(result["byte_budget"]["packed_byte_budget"])
    no_inter_scratch_candidate_abi = bool(result["byte_budget"]["no_inter_scratch_candidate_abi"])
    numeric_ok = (
        reference_error is None
        and candidate_error is None
        and reference_finite
        and candidate_finite
        and rel_l2 is not None
        and max_abs is not None
        and rel_l2 <= args.rel_l2_threshold
        and max_abs <= args.max_abs_threshold
    )
    result["passes"] = {
        "all": bool(numeric_ok and zero_shadow_candidate_abi and packed_byte_budget and no_inter_scratch_candidate_abi),
        "extension_loaded": True,
        "symbol_present": result["symbol_present"],
        "reference_symbol_present": result["reference_symbol_present"],
        "reference_output_returned": reference_error is None,
        "candidate_output_returned": candidate_error is None,
        "reference_finite": reference_finite,
        "candidate_finite": candidate_finite,
        "numeric_vs_reference": bool(numeric_ok),
        "zero_shadow_candidate_abi": zero_shadow_candidate_abi,
        "packed_byte_budget": packed_byte_budget,
        "no_inter_scratch_candidate_abi": no_inter_scratch_candidate_abi,
    }
    result["banked_single_cta_numeric_preflight"] = bool(result["passes"]["all"])
    result["decision"] = "PASS_P4B_SINGLE_CTA_NUMERIC_REFERENCE" if result["passes"]["all"] else "FAIL_P4B_SINGLE_CTA_NUMERIC_REFERENCE"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="P4B single-CTA numeric preflight.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--tile-tokens", type=int, default=1)
    ap.add_argument("--tile-inter", type=int, default=8)
    ap.add_argument("--tile-experts", type=int, default=1)
    ap.add_argument("--tile-hidden", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--rel-l2-threshold", type=float, default=0.05)
    ap.add_argument("--max-abs-threshold", type=float, default=0.5)
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--strict-exit", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.tokens != 1 or args.top_k != 8:
        raise SystemExit("P4B single-CTA numeric preflight currently requires --tokens 1 --top-k 8")
    if args.experts < args.top_k:
        raise SystemExit("--experts must be >= --top-k")

    started = time.time()
    result = run_preflight(args)
    result["elapsed_s"] = round(time.time() - started, 3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[p4b-single-cta-numeric] decision={result.get('decision')}")
    print(f"[p4b-single-cta-numeric] out={out}")
    if result.get("passes", {}).get("all"):
        return 0
    return 2 if args.strict_exit else 0


if __name__ == "__main__":
    raise SystemExit(main())
