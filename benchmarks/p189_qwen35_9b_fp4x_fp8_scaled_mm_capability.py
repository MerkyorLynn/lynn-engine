#!/usr/bin/env python3
"""P189 · R6000 torch._scaled_mm FP4xFP8 capability probe.

Qwen3.5-9B quality gates made W4A8 interesting, but the R6000 fast path cannot
be guessed from high-level PyTorch alone.  This probe records what
`torch._scaled_mm` can actually launch today:

* FP4 x FP4 works through PyTorch, but it is W4A4 and therefore not the target.
* FP8 x FP4 is the desired W4A8 tensor-core shape, but PyTorch 2.10/CUDA 12.8
  does not expose a usable `_scaled_mm` ABI for it.
* The installed CUTLASS/CuTe headers do expose the SM120 `e4m3.e2m1` MMA
  instruction, so the next path is a Lynn-owned CuTe/inline-asm boundary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.native_cuda import discover_native_include_paths  # noqa: E402
from engine.nvfp4_runtime import _compact_scale_to_swizzled_fp8  # noqa: E402
from triton_kernels.nvfp4_linear import quantize_fp4_m1_native  # noqa: E402


def _attempt(name: str, fn) -> dict[str, Any]:
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        out = fn()
        torch.cuda.synchronize()
        return {
            "name": name,
            "ok": True,
            "elapsed_ms": (time.time() - t0) * 1000.0,
            "shape": list(out.shape),
            "dtype": str(out.dtype),
            "mean_abs": float(out.float().abs().mean()),
        }
    except Exception as exc:  # noqa: BLE001 - probe must capture the precise blocker.
        torch.cuda.synchronize()
        return {
            "name": name,
            "ok": False,
            "elapsed_ms": (time.time() - t0) * 1000.0,
            "error_type": type(exc).__name__,
            "error": str(exc).splitlines()[0],
        }


def _header_has_mixed_mma(include_paths: list[str]) -> bool:
    needles = (
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32",
        "SM120_16x8x32_TN<float_e4m3_t, float_e2m1_t, float>",
    )
    for root in include_paths:
        header = Path(root) / "cute" / "arch" / "mma_sm120.hpp"
        if not header.exists():
            continue
        text = header.read_text(encoding="utf-8", errors="ignore")
        if all(needle in text for needle in needles):
            return True
    return False


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("P189 requires CUDA")
    if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("P189 requires torch.float4_e2m1fn_x2 and torch._scaled_mm")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    k = int(args.k)
    n = int(args.n)
    x = torch.randn((1, k), device=device, dtype=torch.bfloat16)
    scale_a_fp8 = (x.float().abs().amax().clamp_min(1.0e-8) / 448.0).to(torch.float32)
    x_fp8 = (x.float() / scale_a_fp8).to(torch.float8_e4m3fn)
    x_fp4_packed, x_fp4_scale = quantize_fp4_m1_native(x)

    # Native NVFP4 checkpoint layout: rows are output channels, columns pack two
    # E2M1 values per byte.  Its transpose is what PyTorch accepts for FP4xFP4.
    w_u8_n_khalf = torch.empty((n, k // 2), device=device, dtype=torch.uint8).random_(0, 255)
    w_fp4_t_natural = w_u8_n_khalf.view(torch.float4_e2m1fn_x2).t()
    scale_b_fp4 = _compact_scale_to_swizzled_fp8(
        torch.ones((n, k // 16), device=device, dtype=torch.float32),
        outer_dim=n,
        k=k,
    )

    # A deliberately expanded storage variant gives B a visible logical K equal
    # to the FP8 A operand.  It still fails today because PyTorch's FP4 scale
    # ABI does not expose this mixed operand configuration.
    w_u8_n_k = torch.empty((n, k), device=device, dtype=torch.uint8).random_(0, 255)
    w_fp4_t_expanded = w_u8_n_k.view(torch.float4_e2m1fn_x2).t()

    w_fp8_nk = torch.randn((n, k), device=device, dtype=torch.float32).to(torch.float8_e4m3fn).contiguous()
    scale_b_fp8 = torch.tensor(1.0, device=device, dtype=torch.float32)

    attempts = [
        _attempt(
            "fp4_x_fp4_control",
            lambda: torch._scaled_mm(
                x_fp4_packed.view(torch.float4_e2m1fn_x2),
                w_fp4_t_natural,
                scale_a=x_fp4_scale,
                scale_b=scale_b_fp4,
                out_dtype=torch.bfloat16,
            ),
        ),
        _attempt(
            "fp8_x_fp8_control",
            lambda: torch._scaled_mm(
                x_fp8,
                w_fp8_nk.t(),
                scale_a=scale_a_fp8,
                scale_b=scale_b_fp8,
                out_dtype=torch.bfloat16,
            ),
        ),
        _attempt(
            "fp8_x_fp4_natural_checkpoint_layout",
            lambda: torch._scaled_mm(
                x_fp8,
                w_fp4_t_natural,
                scale_a=scale_a_fp8,
                scale_b=scale_b_fp4,
                out_dtype=torch.bfloat16,
            ),
        ),
        _attempt(
            "fp8_x_fp4_expanded_logical_k_layout",
            lambda: torch._scaled_mm(
                x_fp8,
                w_fp4_t_expanded,
                scale_a=scale_a_fp8,
                scale_b=scale_b_fp4,
                out_dtype=torch.bfloat16,
            ),
        ),
    ]

    include_paths = discover_native_include_paths()
    mixed_header = _header_has_mixed_mma(include_paths)
    mixed_ok = any(row["name"].startswith("fp8_x_fp4") and row["ok"] for row in attempts)
    fp4_control_ok = any(row["name"] == "fp4_x_fp4_control" and row["ok"] for row in attempts)
    fp8_control_ok = any(row["name"] == "fp8_x_fp8_control" and row["ok"] for row in attempts)
    decision = (
        "TORCH_MIXED_FP4XFP8_AVAILABLE"
        if mixed_ok
        else "TORCH_MIXED_FP4XFP8_UNAVAILABLE_CUTE_REQUIRED"
        if fp4_control_ok and fp8_control_ok and mixed_header
        else "FP4_FP8_TOOLCHAIN_INCOMPLETE"
    )
    return {
        "schema": "lynn-qwen35-9b-p189-fp4x-fp8-scaled-mm-capability-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "k": k,
        "n": n,
        "include_paths": include_paths,
        "cute_sm120_e4m3_e2m1_header": mixed_header,
        "attempts": attempts,
        "decision": decision,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe torch._scaled_mm mixed FP8xFP4 capability on R6000.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=189)
    args = ap.parse_args()
    report = run_probe(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["decision"] != "FP4_FP8_TOOLCHAIN_INCOMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
