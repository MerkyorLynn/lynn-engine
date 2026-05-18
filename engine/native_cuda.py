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
import sysconfig

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]
_EXTENSION = None


def discover_native_include_paths() -> list[str]:
    """Find optional CUDA/CuTe/CUTLASS include roots for native kernels.

    The scalar bridge does not need these headers, but the grouped per-16 FP4
    kernel line does. Keep discovery opt-in/fail-soft so machines without
    deep_gemm/CUTLASS still build the existing extension.
    """
    candidates: list[Path] = []
    for env_name in (
        "LYNN_NATIVE_EXTRA_INCLUDE_DIRS",
        "LYNN_CUTLASS_INCLUDE_DIR",
        "LYNN_DEEP_GEMM_INCLUDE_DIR",
    ):
        raw = os.environ.get(env_name, "")
        for part in raw.split(os.pathsep):
            if part:
                candidates.append(Path(part).expanduser())

    site_roots = [
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        Path(sysconfig.get_paths().get("purelib", "")),
        Path(sysconfig.get_paths().get("platlib", "")),
    ]
    for site_root in site_roots:
        if site_root:
            candidates.append(site_root / "deep_gemm" / "include")

    # R6000 keeps the eval environment separate from the base miniconda env
    # where deep_gemm is installed. Search the common conda roots explicitly so
    # CUTLASS/CuTe can be discovered without forcing the runtime env to import
    # deep_gemm as a Python package.
    for root in (
        Path.home() / "miniconda3",
        Path("/root/miniconda3"),
        Path("/root/autodl-tmp/conda-envs"),
    ):
        for path in root.glob("lib/python*/site-packages/deep_gemm/include"):
            candidates.append(path)
        for path in root.glob("*/lib/python*/site-packages/deep_gemm/include"):
            candidates.append(path)

    out: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        if not path.exists():
            continue
        resolved = str(path.resolve())
        if resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return out


def native_cuda_arch_flags() -> list[str]:
    """Return optional architecture flags for Lynn native CUDA extensions.

    PyTorch's default extension path targets the visible device capability. On
    Blackwell R6000 this means `sm_120`, but P78-P80 proved that E2M1 FP4 MMA
    requires the feature target `sm_120a`. Keep this explicit and opt-in so the
    existing scalar kernels remain portable on machines where `sm_120a` is not
    supported by the installed CUDA stack.
    """
    raw = os.environ.get("LYNN_NATIVE_CUDA_ARCH", "").strip()
    if raw:
        return [f"-arch={raw}"]

    if os.environ.get("LYNN_NATIVE_CUDA_ARCH_AUTO", "0") != "1":
        return []

    if not torch.cuda.is_available():
        return []

    capability = torch.cuda.get_device_capability(0)
    if capability == (12, 0):
        return ["-arch=sm_120a"]
    return []


def native_cuda_extra_cuda_cflags() -> list[str]:
    """Common CUDA compiler flags for Lynn native extensions."""
    arch_flags = native_cuda_arch_flags()
    flags = ["-O3", "--use_fast_math", *arch_flags]
    if os.environ.get("LYNN_ENABLE_SM120A_FP4_MMA", "0") == "1" or any("sm_120a" in flag for flag in arch_flags):
        flags.append("-DLYNN_ENABLE_SM120A_FP4_MMA=1")
    return flags


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
    for extra_bin in (Path.home() / "miniconda3" / "bin", Path("/root/miniconda3/bin")):
        if extra_bin.exists():
            os.environ["PATH"] = f"{extra_bin}:{os.environ.get('PATH', '')}"
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
            str(ROOT / "csrc" / "lynn_native" / "moe_output_owned_bf16.cu"),
            str(ROOT / "csrc" / "lynn_native" / "moe_slot_strict_bf16.cu"),
            str(ROOT / "csrc" / "lynn_native" / "moe_slot_tensorcore_probe.cu"),
            str(ROOT / "csrc" / "lynn_native" / "moe_slot_packed_nvfp4_probe.cu"),
        ],
        build_directory=str(build_root),
        extra_cflags=["-O3"],
        extra_cuda_cflags=native_cuda_extra_cuda_cflags(),
        extra_include_paths=discover_native_include_paths(),
        verbose=verbose,
    )
    return _EXTENSION


__all__ = [
    "discover_native_include_paths",
    "load_lynn_native_extension",
    "native_cuda_arch_flags",
    "native_cuda_extra_cuda_cflags",
]
