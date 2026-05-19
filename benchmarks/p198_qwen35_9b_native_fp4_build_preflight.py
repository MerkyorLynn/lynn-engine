#!/usr/bin/env python3
"""P198 · Qwen3.5-9B native FP4×FP8 build preflight.

Fail-loud preflight that separates build / symbol / runtime capability issues
before blaming P190 for "FP4 MMA not available".

Decision chain:
  BLOCKED_COMPILE        — extension build failed (compile/link error)
  BLOCKED_SYMBOL_MISSING — extension loaded but FP4 MMA probe symbol missing
  BLOCKED_PROBE_FAIL     — probe symbol exists but tiny-tensor smoke failed
  READY_FOR_P190         — all checks pass

Requires R6000 GPU (SM120, CUDA 12.8).  Does NOT modify resident_runner,
server, or engine code.
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

# ── FP4 MMA symbols expected in the extension ─────────────────────────────────
_FP4_MMA_SYMBOLS = [
    "dense_fp4xfp8_mma_scaled_probe",
    "dense_fp4xfp8_mma_probe",
    "dense_fp4xfp8_scalar_reference",
]


def _probe_env(stamp: str) -> dict[str, str]:
    """Return env overlay for FP4 MMA build."""
    return {
        "LYNN_ENABLE_SM120A_FP4_MMA": "1",
        "LYNN_NATIVE_CUDA_ARCH_AUTO": "1",
        "LYNN_NATIVE_CUDA_BUILD_DIR": f"/tmp/lynn_engine_native_build/p198_{stamp}",
    }


def _env_snapshot() -> dict[str, Any]:
    """Capture runtime environment for the report."""
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
    return info


def _check_cute_headers() -> dict[str, Any]:
    """Check whether CuTe SM120 headers are discoverable."""
    from engine.native_cuda import discover_native_include_paths

    paths = discover_native_include_paths()
    sm120_found = False
    for p in paths:
        candidate = Path(p) / "cute" / "arch" / "mma_sm120.hpp"
        if candidate.exists():
            sm120_found = True
            break
    return {"cute_header_found": sm120_found, "cute_include_paths": paths}


def _run_preflight(stamp: str) -> dict[str, Any]:
    """Execute the full preflight chain."""
    result: dict[str, Any] = {
        "stamp": stamp,
        **_env_snapshot(),
        **_check_cute_headers(),
    }

    build_dir = f"/tmp/lynn_engine_native_build/p198_{stamp}"
    result["build_dir"] = build_dir

    # ── Step 1: try to build + load the extension ──────────────────────────
    ext = None
    load_error: str | None = None
    try:
        from engine.native_cuda import load_lynn_native_extension
        ext = load_lynn_native_extension(build_dir=build_dir, verbose=False)
        result["extension_loaded"] = True
    except Exception as exc:
        result["extension_loaded"] = False
        load_error = str(exc)
        # Keep the tail for the report (last 2000 chars)
        result["load_error_tail"] = load_error[-2000:] if load_error else ""
        result["extension_path"] = None
        result["available_symbols"] = []
        result["probe_result"] = "skipped"
        result["decision"] = "BLOCKED_COMPILE"
        return result

    # ── Step 2: enumerate available symbols ────────────────────────────────
    available = [s for s in _FP4_MMA_SYMBOLS if hasattr(ext, s)]
    missing = [s for s in _FP4_MMA_SYMBOLS if not hasattr(ext, s)]
    result["available_symbols"] = available
    result["missing_symbols"] = missing

    # Try to locate the .so path
    ext_mod = sys.modules.get("lynn_native_runtime")
    result["extension_path"] = getattr(ext_mod, "__file__", None) if ext_mod else None

    if missing:
        result["probe_result"] = "skipped"
        result["decision"] = "BLOCKED_SYMBOL_MISSING"
        return result

    # ── Step 3: run tiny-tensor smoke through the real MMA probe ───────────
    try:
        K = 32
        N = 8
        act_fp8 = torch.randint(0, 255, (K,), dtype=torch.uint8, device="cuda")
        act_scale = torch.ones(K // 16, dtype=torch.float32, device="cuda")
        weight_packed = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device="cuda")
        weight_scale = torch.ones(N, K // 16, dtype=torch.float32, device="cuda")
        weight_global = torch.tensor(1.0, dtype=torch.float32, device="cuda")

        out = ext.dense_fp4xfp8_mma_scaled_probe(
            act_fp8, act_scale, weight_packed, weight_scale, weight_global,
            1, N, K,
        )
        torch.cuda.synchronize()

        result["probe_result"] = "pass"
        result["probe_output_shape"] = list(out.shape)
        result["probe_output_dtype"] = str(out.dtype)
        result["decision"] = "READY_FOR_P190"
    except Exception as exc:
        result["probe_result"] = "fail"
        result["probe_error"] = str(exc)[-1000:]
        result["decision"] = "BLOCKED_PROBE_FAIL"

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="P198 native FP4×FP8 build preflight.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--build-dir", default=None, help="Override build dir.")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")

    # Set env overlay
    overlay = _probe_env(stamp)
    old_env: dict[str, str | None] = {}
    for k, v in overlay.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = v

    print(f"[p198] stamp={stamp}")
    print(f"[p198] env: LYNN_ENABLE_SM120A_FP4_MMA=1")
    print(f"[p198] env: LYNN_NATIVE_CUDA_ARCH_AUTO=1")
    print(f"[p198] env: LYNN_NATIVE_CUDA_BUILD_DIR={overlay['LYNN_NATIVE_CUDA_BUILD_DIR']}")
    print(f"[p198] torch={torch.__version__}  cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"[p198] device={torch.cuda.get_device_name(0)}  capability={cap}")
    print()

    t0 = time.time()
    result = _run_preflight(stamp)
    elapsed = round(time.time() - t0, 1)
    result["elapsed_s"] = elapsed

    # Restore env
    for k, v in old_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    # Pretty print
    decision = result["decision"]
    icon = {
        "READY_FOR_P190": "✅",
        "BLOCKED_COMPILE": "❌",
        "BLOCKED_SYMBOL_MISSING": "⚠️",
        "BLOCKED_PROBE_FAIL": "❌",
    }.get(decision, "❓")

    print(f"[p198] decision: {icon} {decision}")
    print(f"[p198] elapsed: {elapsed}s")
    print()

    if decision == "BLOCKED_COMPILE":
        print("[p198] === BUILD ERROR TAIL ===")
        print(result.get("load_error_tail", "(empty)"))
        print("[p198] === END ===")
    elif decision == "BLOCKED_SYMBOL_MISSING":
        print(f"[p198] missing: {result.get('missing_symbols', [])}")
        print(f"[p198] available: {result.get('available_symbols', [])}")
    elif decision == "BLOCKED_PROBE_FAIL":
        print(f"[p198] probe error: {result.get('probe_error', '')}")
    else:
        print(f"[p198] probe output: shape={result.get('probe_output_shape')} dtype={result.get('probe_output_dtype')}")
        print(f"[p198] extension: {result.get('extension_path', 'N/A')}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[p198] report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
