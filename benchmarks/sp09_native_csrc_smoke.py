#!/usr/bin/env python3
"""SP-09: Lynn native CUDA extension build smoke on Spark sm_121.

Verifies:
  1. csrc/lynn_native/{bindings.cpp, moe_scalar_kernel.cu, smoke_kernel.cu}
     compile on sm_121 (Spark GB10) under CUDA 13.0 / torch 2.9.1.
  2. The smoke kernel `add_one` actually runs (CUDA path live).
  3. The P65 grouped per-16 ABI contract function is callable and produces
     the expected guard message.

Run on Spark inside the SGLang dev-cu13 docker container. Build artifacts
land in /tmp/lynn_engine_native_build/runtime/.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch


def main() -> int:
    print(f"[sp09] python: {sys.version.split()[0]}")
    print(f"[sp09] torch:  {torch.__version__}")
    print(f"[sp09] cuda:   {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("[sp09] FAIL: CUDA not available")
        return 1
    cap = torch.cuda.get_device_capability(0)
    print(f"[sp09] device: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
    print()

    arch_override = os.environ.get("LYNN_NATIVE_CUDA_ARCH", "<not set>")
    print(f"[sp09] LYNN_NATIVE_CUDA_ARCH={arch_override}")
    if arch_override == "<not set>":
        print("[sp09] hint: try LYNN_NATIVE_CUDA_ARCH=sm_121a or sm_121")
    print()

    print("[sp09] [1/3] loading Lynn native extension (JIT build via nvcc)...")
    t0 = time.time()
    try:
        from engine.native_cuda import load_lynn_native_extension
        ext = load_lynn_native_extension(verbose=True)
    except Exception as exc:
        print(f"[sp09] FAIL build: {type(exc).__name__}: {exc}")
        return 2
    print(f"[sp09] build OK in {time.time() - t0:.1f}s")
    print()

    print("[sp09] [2/3] smoke kernel add_one")
    try:
        x = torch.zeros(8, device="cuda", dtype=torch.float32)
        y = ext.add_one(x)
        ok = bool(torch.all(y == 1.0).item())
        print(f"[sp09] add_one(zeros(8)) = {y.tolist()}  ok={ok}")
        if not ok:
            print("[sp09] FAIL: add_one did not produce 1.0")
            return 3
    except Exception as exc:
        print(f"[sp09] FAIL smoke: {type(exc).__name__}: {exc}")
        return 4
    print()

    print("[sp09] [3/3] P65 grouped per-16 active-MoE contract guard")
    if not hasattr(ext, "active_moe_grouped_per16_contract"):
        print("[sp09] WARN: active_moe_grouped_per16_contract not exposed in this csrc snapshot")
        print(f"[sp09] available functions: {[n for n in dir(ext) if not n.startswith('_')][:20]}")
        print("[sp09] this can mean we extracted from a stale point — re-check csrc/lynn_native/bindings.cpp")
        return 5

    # Construct dummy tensors with the right shapes/dtypes — the guard does
    # shape checks, not real math.
    HIDDEN = 2048
    INTERMEDIATE = 512
    TOP_K = 8
    EXPERTS = 256
    device = "cuda"

    x = torch.zeros(HIDDEN, device=device, dtype=torch.bfloat16)
    expert_ids = torch.zeros(TOP_K, device=device, dtype=torch.int32)
    routing_weights = torch.ones(TOP_K, device=device, dtype=torch.float32) / TOP_K
    gate_up_packed = torch.zeros(EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2, device=device, dtype=torch.uint8)
    gate_up_scale = torch.ones(EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16, device=device, dtype=torch.float32)
    gate_up_global = torch.ones(1, device=device, dtype=torch.float32)
    down_packed = torch.zeros(EXPERTS, HIDDEN, INTERMEDIATE // 2, device=device, dtype=torch.uint8)
    down_scale = torch.ones(EXPERTS, HIDDEN, INTERMEDIATE // 16, device=device, dtype=torch.float32)
    down_global = torch.ones(1, device=device, dtype=torch.float32)

    try:
        out = ext.active_moe_grouped_per16_contract(
            x, expert_ids, routing_weights,
            gate_up_packed, gate_up_scale, gate_up_global,
            down_packed, down_scale, down_global,
        )
        # Codex's P65 doc says this is expected to RAISE with a guard message
        # because the kernel isn't implemented yet. If it returns silently,
        # something has changed since P65.
        print(f"[sp09] UNEXPECTED success: out shape {out.shape} dtype {out.dtype}")
        print("[sp09] (P65 said this should still raise 'not implemented'; csrc must have been updated)")
    except RuntimeError as exc:
        msg = str(exc)
        if "not implemented" in msg or "grouped per-16" in msg:
            print(f"[sp09] PASS: shape/layout guard accepted inputs, raised expected unimplemented message:")
            print(f"[sp09]   {msg[:200]}")
        else:
            print(f"[sp09] FAIL: contract failed with unexpected error:")
            print(f"[sp09]   {msg[:300]}")
            return 6

    print()
    print("[sp09] === DONE ===")
    print("[sp09] csrc compiles + smoke runs + P65 ABI guards correctly on Spark sm_121")
    return 0


if __name__ == "__main__":
    sys.exit(main())
