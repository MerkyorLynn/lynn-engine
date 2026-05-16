#!/usr/bin/env python3
"""P78: execute one sm_120 E2M1xE2M1 FP4 MMA instruction.

P76 verifies headers/toolchain. P77 verifies Lynn nibble encoding is compatible
with CUTLASS E2M1 storage. P78 is the next smallest gate: actually call the
Blackwell FP4 MMA wrapper from a torch CUDA extension.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.native_cuda import discover_native_include_paths  # noqa: E402


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor p78_sm120_e2m1_mma_smoke();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sm120_e2m1_mma_smoke", &p78_sm120_e2m1_mma_smoke, "P78 sm120 E2M1 MMA smoke");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace {

__global__ void p78_mma_kernel(float* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    // All-zero E2M1 registers encode zero A/B operands. The MMA result should
    // therefore preserve the C accumulator values exactly.
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

torch::Tensor p78_sm120_e2m1_mma_smoke() {
  auto out = torch::empty({4}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  p78_mma_kernel<<<1, 32>>>(out.data_ptr<float>());
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "p78_mma_kernel launch failed: ", cudaGetErrorString(err));
  return out;
}
"""


def _prepare_path() -> None:
    python_bin = Path(sys.executable).resolve().parent
    os.environ["PATH"] = f"{python_bin}:{os.environ.get('PATH', '')}"
    for extra_bin in (Path.home() / "miniconda3" / "bin", Path("/root/miniconda3/bin")):
        if extra_bin.exists():
            os.environ["PATH"] = f"{extra_bin}:{os.environ.get('PATH', '')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p78_sm120_e2m1_mma")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P78 requires CUDA")

    _prepare_path()
    include_paths = discover_native_include_paths()
    build_root = Path(args.build_dir)
    build_root.mkdir(parents=True, exist_ok=True)
    cpp_path = build_root / "p78_sm120_mma_bindings.cpp"
    cu_path = build_root / "p78_sm120_mma_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")

    t0 = time.time()
    compile_ok = False
    error: str | None = None
    values: list[float] | None = None
    try:
        module = load(
            name="lynn_native_p78_sm120_e2m1_mma_smoke",
            sources=[str(cpp_path), str(cu_path)],
            build_directory=str(build_root),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_include_paths=include_paths,
            verbose=args.verbose,
        )
        out = module.sm120_e2m1_mma_smoke()
        torch.cuda.synchronize()
        values = [float(x) for x in out.cpu().tolist()]
        compile_ok = values == [1.0, 2.0, 3.0, 4.0]
    except Exception as exc:  # noqa: BLE001 - preserve the native compile/runtime error.
        error = repr(exc)
    build_s = time.time() - t0

    result = {
        "schema_version": "lynn-engine-p78-sm120-e2m1-mma-smoke-v1",
        "torch": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "include_paths": include_paths,
        "compile_and_run_ok": compile_ok,
        "build_seconds": build_s,
        "values": values,
        "expected_values": [1.0, 2.0, 3.0, 4.0],
        "error": error,
        "decision": (
            "sm_120 E2M1xE2M1 FP4 MMA instruction path executes inside the Lynn torch extension."
            if compile_ok
            else "sm_120 E2M1xE2M1 FP4 MMA smoke failed; inspect compiler/runtime error before writing the grouped kernel."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if compile_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
