#!/usr/bin/env python3
"""Stage 6 P4 native fused-MoE ABI preflight.

This is a capability/contract gate, not a speed gate. It verifies that the
caller-owned-output packed-NVFP4 zero-shadow C++ hot-path symbol is built into
the Lynn native extension and that a valid tensor bundle reaches the intentional
P4 "kernel not implemented yet" fail-loud boundary.
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

SYMBOL = "active_moe_fused_zero_shadow_out_contract"
EXPECTED_ERROR = "P4 fused 4-bit zero-shadow CUDA kernel is not implemented yet"


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


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "lynn-stage6-p4-native-fused-moe-abi-preflight-v1",
        "symbol": SYMBOL,
        "expected_error": EXPECTED_ERROR,
        "tokens": args.tokens,
        "experts": args.experts,
        "top_k": args.top_k,
        "tile_tokens": args.tile_tokens,
        "tile_inter": args.tile_inter,
        "tile_hidden": args.tile_hidden,
        "banked_native_abi_preflight": False,
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
        f"/tmp/lynn_engine_native_build/p4_native_abi_{time.strftime('%Y%m%d_%H%M%S')}",
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
            args.tile_inter,
            args.tile_hidden,
        )
        torch.cuda.synchronize()
        result["decision"] = "UNEXPECTED_IMPLEMENTED"
        result["passes"] = {
            "all": False,
            "extension_loaded": True,
            "symbol_present": True,
            "fail_loud_boundary": False,
        }
    except Exception as exc:
        message = str(exc)
        result["call_error_tail"] = message[-2000:]
        fail_loud = EXPECTED_ERROR in message
        result["decision"] = "PASS_ABI_CONTRACT" if fail_loud else "BLOCKED_GUARD_OR_RUNTIME"
        result["banked_native_abi_preflight"] = bool(fail_loud)
        result["passes"] = {
            "all": bool(fail_loud),
            "extension_loaded": True,
            "symbol_present": True,
            "fail_loud_boundary": bool(fail_loud),
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="P4 native fused-MoE ABI preflight.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--tokens", type=int, default=2)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--tile-tokens", type=int, default=1)
    ap.add_argument("--tile-inter", type=int, default=8)
    ap.add_argument("--tile-hidden", type=int, default=8)
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--strict-exit", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.tokens <= 0 or args.experts <= 0 or args.top_k <= 0:
        raise SystemExit("--tokens, --experts, and --top-k must be positive")
    if args.top_k > args.experts:
        raise SystemExit("--top-k must be <= --experts for the ABI fixture")

    started = time.time()
    result = run_preflight(args)
    result["elapsed_s"] = round(time.time() - started, 3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    decision = result.get("decision")
    print(f"[p4-native-abi] decision={decision}")
    print(f"[p4-native-abi] out={out}")
    if result.get("passes", {}).get("all"):
        return 0
    return 2 if args.strict_exit else 0


if __name__ == "__main__":
    raise SystemExit(main())
