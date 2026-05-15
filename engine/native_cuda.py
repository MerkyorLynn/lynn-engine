"""Lynn-owned CUDA extension loader.

The extension is intentionally opt-in. P27-P30 established build/load and
active-MoE contract correctness; production still defaults to the faster Triton
path until the native kernels replace scalar inner loops with true grouped FP4
math.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]
_EXTENSION = None


def load_lynn_native_extension(*, build_dir: str | None = None, verbose: bool = False):
    """Build/load the Lynn native CUDA extension once per Python process."""
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not torch.cuda.is_available():
        raise RuntimeError("Lynn native CUDA extension requires CUDA")

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", ".".join(map(str, torch.cuda.get_device_capability(0))))
    python_bin = Path(sys.executable).resolve().parent
    os.environ["PATH"] = f"{python_bin}:{os.environ.get('PATH', '')}"
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    nvcc_bin = Path(cuda_home) / "bin"
    if (nvcc_bin / "nvcc").exists():
        os.environ["PATH"] = f"{nvcc_bin}:{os.environ.get('PATH', '')}"

    build_root = Path(build_dir or os.environ.get("LYNN_NATIVE_CUDA_BUILD_DIR", "/tmp/lynn_engine_native_build/runtime"))
    build_root.mkdir(parents=True, exist_ok=True)
    _EXTENSION = load(
        name="lynn_native_runtime",
        sources=[
            str(ROOT / "csrc" / "lynn_native" / "bindings.cpp"),
            str(ROOT / "csrc" / "lynn_native" / "smoke_kernel.cu"),
            str(ROOT / "csrc" / "lynn_native" / "moe_scalar_kernel.cu"),
        ],
        build_directory=str(build_root),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=verbose,
    )
    return _EXTENSION


__all__ = ["load_lynn_native_extension"]
