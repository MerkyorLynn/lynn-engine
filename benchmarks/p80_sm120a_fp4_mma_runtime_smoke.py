#!/usr/bin/env python3
"""P80: force feature-target FP4 MMA through a torch extension.

P79 showed that `sm_120a` accepts the CuTe E2M1 MMA source while default
`sm_120` rejects it. P80 checks whether a Lynn torch extension can force that
target and actually execute the instruction on the R6000.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.native_cuda import discover_native_include_paths  # noqa: E402


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor p80_fp4_mma_smoke();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_mma_smoke", &p80_fp4_mma_smoke, "P80 forced-target E2M1 MMA smoke");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace {

__global__ void p80_mma_kernel(float* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    const uint32_t a0 = 0u;
    const uint32_t a1 = 0u;
    const uint32_t a2 = 0u;
    const uint32_t a3 = 0u;
    const uint32_t b0 = 0u;
    const uint32_t b1 = 0u;
    const float c0 = 1.0f;
    const float c1 = 2.0f;
    const float c2 = 3.0f;
    const float c3 = 4.0f;

    cute::SM120_16x8x32_TN<
        cute::float_e2m1_t,
        cute::float_e2m1_t,
        float>::fma(
            d0, d1, d2, d3,
            a0, a1, a2, a3,
            b0, b1,
            c0, c1, c2, c3);

    out[0] = d0;
    out[1] = d1;
    out[2] = d2;
    out[3] = d3;
  }
}

}  // namespace

torch::Tensor p80_fp4_mma_smoke() {
  auto out = torch::empty({4}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  p80_mma_kernel<<<1, 32>>>(out.data_ptr<float>());
  const cudaError_t launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess, "p80_mma_kernel launch failed: ", cudaGetErrorString(launch_err));
  const cudaError_t sync_err = cudaDeviceSynchronize();
  TORCH_CHECK(sync_err == cudaSuccess, "p80_mma_kernel sync failed: ", cudaGetErrorString(sync_err));
  return out;
}
"""


VARIANTS = [
    {
        "name": "arch_sm120a",
        "extra_cuda_cflags": ["-O3", "--use_fast_math", "-arch=sm_120a"],
    },
    {
        "name": "gencode_sm120a",
        "extra_cuda_cflags": ["-O3", "--use_fast_math", "-gencode=arch=compute_120a,code=sm_120a"],
    },
    {
        "name": "arch_compute120a",
        "extra_cuda_cflags": ["-O3", "--use_fast_math", "-arch=compute_120a"],
    },
    {
        "name": "arch_compute120",
        "extra_cuda_cflags": ["-O3", "--use_fast_math", "-arch=compute_120"],
    },
]


def _prepare_path() -> None:
    python_bin = Path(sys.executable).resolve().parent
    os.environ["PATH"] = f"{python_bin}:{os.environ.get('PATH', '')}"
    for extra_bin in (
        Path.home() / "miniconda3" / "bin",
        Path("/root/miniconda3/bin"),
        Path("/usr/local/cuda/bin"),
    ):
        if extra_bin.exists():
            os.environ["PATH"] = f"{extra_bin}:{os.environ.get('PATH', '')}"


def _write_sources(build_root: Path) -> tuple[Path, Path]:
    build_root.mkdir(parents=True, exist_ok=True)
    cpp_path = build_root / "p80_bindings.cpp"
    cu_path = build_root / "p80_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")
    return cpp_path, cu_path


def _try_variant(
    variant: dict[str, object],
    build_root: Path,
    include_paths: list[str],
    verbose: bool,
) -> dict[str, object]:
    name = str(variant["name"])
    variant_root = build_root / name
    if variant_root.exists():
        shutil.rmtree(variant_root)
    cpp_path, cu_path = _write_sources(variant_root)

    t0 = time.time()
    values: list[float] | None = None
    error: str | None = None
    try:
        # PyTorch skips its default CUDA arch flags when extra flags already
        # contain an arch/gencode selector. That lets P80 avoid the failing
        # default `sm_120` flag path proven by P78.
        module = load(
            name=f"lynn_native_p80_{name}",
            sources=[str(cpp_path), str(cu_path)],
            build_directory=str(variant_root),
            extra_cflags=["-O3"],
            extra_cuda_cflags=list(variant["extra_cuda_cflags"]),
            extra_include_paths=include_paths,
            verbose=verbose,
        )
        out = module.fp4_mma_smoke()
        torch.cuda.synchronize()
        values = [float(x) for x in out.cpu().tolist()]
    except Exception as exc:  # noqa: BLE001 - preserve native build/runtime error.
        error = repr(exc)

    ok = values == [1.0, 2.0, 3.0, 4.0]
    return {
        "name": name,
        "extra_cuda_cflags": variant["extra_cuda_cflags"],
        "ok": ok,
        "seconds": time.time() - t0,
        "values": values,
        "expected_values": [1.0, 2.0, 3.0, 4.0],
        "error": error,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p80_sm120a_fp4_mma_runtime_smoke")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P80 requires CUDA")

    _prepare_path()
    include_paths = discover_native_include_paths()
    build_root = Path(args.build_dir)
    build_root.mkdir(parents=True, exist_ok=True)

    variants = [
        _try_variant(variant, build_root, include_paths, args.verbose)
        for variant in VARIANTS
    ]
    runnable = [variant for variant in variants if variant["ok"]]
    result = {
        "schema_version": "lynn-engine-p80-sm120a-fp4-mma-runtime-smoke-v1",
        "torch": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "include_paths": include_paths,
        "variants": variants,
        "any_runtime_ok": bool(runnable),
        "decision": (
            f"FP4 MMA runtime smoke PASS via {runnable[0]['name']}."
            if runnable
            else "No forced target variant compiled and executed the CuTe E2M1 MMA smoke."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if runnable else 1


if __name__ == "__main__":
    raise SystemExit(main())
