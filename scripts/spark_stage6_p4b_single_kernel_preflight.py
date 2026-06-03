#!/usr/bin/env python3
"""Stage 6 P4B true fused single-kernel preflight.

This is a fail-loud contract gate, not a speed gate. P4B is not implemented
yet; the native symbol is expected to compile, expose the single-kernel ABI,
accept a zero-shadow tensor bundle with no inter_scratch, and then throw the
explicit not-implemented guard. If it returns today, the preflight fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SYMBOL = "active_moe_fused_zero_shadow_single_kernel_contract"
SCHEMA = "lynn-stage6-p4b-single-kernel-preflight-v1"
FAIL_LOUD_NEEDLES = (
    "P4B single-kernel fused zero-shadow contract is not implemented yet",
    "do not bank fused-kernel speed or promote this backend",
)


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


def _make_inputs(tokens: int, experts: int, top_k: int) -> dict[str, torch.Tensor]:
    hidden = torch.zeros(tokens, 2048, dtype=torch.bfloat16, device="cuda")
    expert_ids = torch.zeros(tokens, top_k, dtype=torch.int32, device="cuda")
    routing_weights = torch.full((tokens, top_k), 1.0 / top_k, dtype=torch.float32, device="cuda")
    gate_up_packed = torch.zeros(experts, 1024, 1024, dtype=torch.uint8, device="cuda")
    gate_up_scale = torch.ones(experts, 1024, 128, dtype=torch.float32, device="cuda")
    gate_up_global_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    down_packed = torch.zeros(experts, 2048, 256, dtype=torch.uint8, device="cuda")
    down_scale = torch.ones(experts, 2048, 32, dtype=torch.float32, device="cuda")
    down_global_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    out = torch.empty_like(hidden)
    return {
        "hidden": hidden,
        "expert_ids": expert_ids,
        "routing_weights": routing_weights,
        "gate_up_packed": gate_up_packed,
        "gate_up_scale": gate_up_scale,
        "gate_up_global_scale": gate_up_global_scale,
        "down_packed": down_packed,
        "down_scale": down_scale,
        "down_global_scale": down_global_scale,
        "out": out,
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
    activation_io_names = ["hidden", "expert_ids", "routing_weights", "out"]
    forbidden_shadow_names = [
        "gate_up_proj",
        "down_proj",
        "gate_up_weight",
        "down_weight",
        "inter_scratch",
    ]
    packed_weight_bytes = sum(int((tensor_manifest.get(name) or {}).get("bytes") or 0) for name in packed_weight_names)
    activation_io_bytes = sum(int((tensor_manifest.get(name) or {}).get("bytes") or 0) for name in activation_io_names)
    # Equivalent full BF16 active-expert shadow for this fixture:
    # gate/up [E, 1024, 2048] + down [E, 2048, 512], two bytes per BF16 element.
    bf16_shadow_equivalent_bytes = int(experts * ((1024 * 2048) + (2048 * 512)) * 2)
    forbidden_present = [
        name
        for name in tensor_manifest
        if any(forbidden in name for forbidden in forbidden_shadow_names)
    ]
    return {
        "packed_weight_names": packed_weight_names,
        "activation_io_names": activation_io_names,
        "packed_weight_bytes": packed_weight_bytes,
        "activation_io_bytes": activation_io_bytes,
        "bf16_shadow_equivalent_bytes": bf16_shadow_equivalent_bytes,
        "packed_vs_bf16_shadow_ratio": (
            packed_weight_bytes / bf16_shadow_equivalent_bytes if bf16_shadow_equivalent_bytes else None
        ),
        "forbidden_shadow_tensor_names": forbidden_present,
        "zero_shadow_abi": not forbidden_present,
        "packed_byte_budget": packed_weight_bytes > 0 and packed_weight_bytes < bf16_shadow_equivalent_bytes,
        "no_inter_scratch_abi": "inter_scratch" not in tensor_manifest,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "symbol": SYMBOL,
        "expected_behavior": "fail-loud single-kernel contract; no fused implementation is banked",
        "tokens": args.tokens,
        "experts": args.experts,
        "top_k": args.top_k,
        "tile_tokens": args.tile_tokens,
        "tile_experts": args.tile_experts,
        "tile_hidden": args.tile_hidden,
        "banked_single_kernel_contract_preflight": False,
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
        f"/tmp/lynn_engine_native_build/p4b_single_kernel_{time.strftime('%Y%m%d_%H%M%S')}",
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

    has_symbol = hasattr(ext, SYMBOL)
    result["symbol_present"] = bool(has_symbol)
    if not has_symbol:
        result["decision"] = "BLOCKED_SYMBOL_MISSING"
        result["passes"] = {"all": False, "extension_loaded": True, "symbol_present": False}
        return result

    inputs = _make_inputs(args.tokens, args.experts, args.top_k)
    result["tensor_manifest"] = _tensor_manifest(inputs)
    result["byte_budget"] = _byte_budget(result["tensor_manifest"], experts=args.experts)

    call_returned = False
    call_error = ""
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
            inputs["out"],
            args.tile_tokens,
            args.tile_experts,
            args.tile_hidden,
        )
        torch.cuda.synchronize()
        call_returned = True
    except Exception as exc:
        call_error = str(exc)

    zero_shadow_abi = bool(result["byte_budget"]["zero_shadow_abi"])
    packed_byte_budget = bool(result["byte_budget"]["packed_byte_budget"])
    no_inter_scratch_abi = bool(result["byte_budget"]["no_inter_scratch_abi"])
    fail_loud_not_implemented = bool(call_error) and all(needle in call_error for needle in FAIL_LOUD_NEEDLES)
    result["call_returned"] = call_returned
    if call_error:
        result["call_error_tail"] = call_error[-2000:]
    result["fail_loud_needles"] = list(FAIL_LOUD_NEEDLES)
    result["banked_single_kernel_contract_preflight"] = bool(
        fail_loud_not_implemented and zero_shadow_abi and packed_byte_budget and no_inter_scratch_abi
    )
    result["decision"] = (
        "PASS_SINGLE_KERNEL_FAILLOUD_CONTRACT"
        if result["banked_single_kernel_contract_preflight"]
        else "FAIL_SINGLE_KERNEL_CONTRACT"
    )
    result["passes"] = {
        "all": bool(result["banked_single_kernel_contract_preflight"]),
        "extension_loaded": True,
        "symbol_present": True,
        "call_returned_false": not call_returned,
        "fail_loud_not_implemented": fail_loud_not_implemented,
        "zero_shadow_abi": zero_shadow_abi,
        "packed_byte_budget": packed_byte_budget,
        "no_inter_scratch_abi": no_inter_scratch_abi,
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="P4B single-kernel fail-loud preflight.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--tile-tokens", type=int, default=1)
    ap.add_argument("--tile-experts", type=int, default=1)
    ap.add_argument("--tile-hidden", type=int, default=8)
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--strict-exit", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.tokens <= 0 or args.experts <= 0 or args.top_k <= 0:
        raise SystemExit("--tokens, --experts, and --top-k must be positive")
    if args.top_k > args.experts:
        raise SystemExit("--top-k must be <= --experts for the ABI fixture")
    if args.tile_tokens <= 0 or args.tile_experts <= 0 or args.tile_hidden <= 0:
        raise SystemExit("--tile-* values must be positive")

    started = time.time()
    result = run_preflight(args)
    result["elapsed_s"] = round(time.time() - started, 3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    decision = result.get("decision")
    print(f"[p4b-single-kernel] decision={decision}")
    print(f"[p4b-single-kernel] out={out}")
    if result.get("passes", {}).get("all"):
        return 0
    return 2 if args.strict_exit else 0


if __name__ == "__main__":
    raise SystemExit(main())
