#!/usr/bin/env python3
"""P27: native CUDA extension build/load smoke.

P16-P26 narrowed the 155 TPS gap to a real custom native FP4 active expert
kernel. Before writing that kernel, this gate proves the target machine can
compile and import a Lynn-owned CUDA extension against the active PyTorch/CUDA
stack.

This intentionally builds a tiny `add_one` kernel only. Passing this gate means
the remaining work is kernel math/layout, not repository/build plumbing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys
import time

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p27_smoke")
    ap.add_argument("--size", type=int, default=1_000_000)
    args = ap.parse_args()

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    nvcc = Path(cuda_home) / "bin" / "nvcc"
    python_bin = Path(sys.executable).resolve().parent
    os.environ["PATH"] = f"{python_bin}:{os.environ.get('PATH', '')}"
    if nvcc.exists():
        os.environ["PATH"] = f"{nvcc.parent}:{os.environ.get('PATH', '')}"

    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        str(ROOT / "csrc" / "lynn_native" / "bindings.cpp"),
        str(ROOT / "csrc" / "lynn_native" / "smoke_kernel.cu"),
    ]

    t0 = time.time()
    module = load(
        name="lynn_native_smoke",
        sources=sources,
        build_directory=str(build_dir),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    build_s = time.time() - t0

    x = torch.arange(args.size, device="cuda", dtype=torch.float32)
    y = module.add_one(x)
    torch.cuda.synchronize()
    max_abs = float((y - (x + 1.0)).abs().max().item())

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        y = module.add_one(x)
    torch.cuda.synchronize()
    start.record()
    iters = 100
    for _ in range(iters):
        y = module.add_one(x)
    end.record()
    torch.cuda.synchronize()
    avg_ms = float(start.elapsed_time(end) / iters)

    result = {
        "schema_version": "lynn-engine-p27-cuda-extension-smoke-v1",
        "pass": max_abs == 0.0,
        "root": str(ROOT),
        "sources": sources,
        "build_dir": str(build_dir),
        "build_seconds": build_s,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_home": cuda_home,
        "nvcc": str(nvcc) if nvcc.exists() else shutil.which("nvcc"),
        "nvcc_version": _run([str(nvcc), "--version"]) if nvcc.exists() else _run(["nvcc", "--version"]),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "size": args.size,
        "max_abs": max_abs,
        "avg_ms": avg_ms,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
