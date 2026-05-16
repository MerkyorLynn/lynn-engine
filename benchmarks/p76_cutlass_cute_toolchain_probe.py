#!/usr/bin/env python3
"""P76: CUTLASS/CuTe toolchain smoke for grouped per-16 FP4 work.

P75 closes scalar gate/up launch tuning. Before writing the real grouped
per-16 active-expert kernel, verify that the current R6000 environment can
compile a native extension that sees CUTLASS/CuTe Blackwell FP4 types and
sm_120 MMA headers. This is a toolchain gate, not a performance benchmark.
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

torch::Tensor p76_cutlass_cute_smoke();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cutlass_cute_smoke", &p76_cutlass_cute_smoke, "P76 CUTLASS/CuTe smoke");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cutlass/float8.h>
#include <cutlass/float_subbyte.h>
#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace {

__global__ void p76_smoke_kernel(float* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = static_cast<float>(cutlass::sizeof_bits<cutlass::float_e2m1_t>::value);
    out[1] = static_cast<float>(cutlass::sizeof_bits<cutlass::float_ue8m0_t>::value);
    out[2] = static_cast<float>(sizeof(cute::SM120_16x8x32_TN<
        cute::float_e2m1_t,
        cute::float_e2m1_t,
        float>));
    out[3] = 120.0f;
  }
}

}  // namespace

torch::Tensor p76_cutlass_cute_smoke() {
  auto out = torch::empty({4}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  p76_smoke_kernel<<<1, 32>>>(out.data_ptr<float>());
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "p76_smoke_kernel launch failed: ", cudaGetErrorString(err));
  return out;
}
"""


def _header_exists(include_dir: str, relative: str) -> bool:
    return (Path(include_dir) / relative).exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p76_cutlass_cute")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P76 requires CUDA")

    include_paths = discover_native_include_paths()
    required_headers = [
        "cutlass/float_subbyte.h",
        "cutlass/float8.h",
        "cute/arch/mma_sm120.hpp",
        "cute/numeric/numeric_types.hpp",
    ]
    header_status = {
        header: any(_header_exists(path, header) for path in include_paths)
        for header in required_headers
    }
    missing = [header for header, ok in header_status.items() if not ok]
    compile_attempted = not missing
    compile_ok = False
    smoke_values: list[float] | None = None
    error: str | None = None
    build_s: float | None = None

    if compile_attempted:
        python_bin = Path(sys.executable).resolve().parent
        os.environ["PATH"] = f"{python_bin}:{os.environ.get('PATH', '')}"
        for extra_bin in (Path.home() / "miniconda3" / "bin", Path("/root/miniconda3/bin")):
            if extra_bin.exists():
                os.environ["PATH"] = f"{extra_bin}:{os.environ.get('PATH', '')}"
        build_root = Path(args.build_dir)
        build_root.mkdir(parents=True, exist_ok=True)
        cpp_path = build_root / "p76_cutlass_cute_bindings.cpp"
        cu_path = build_root / "p76_cutlass_cute_kernel.cu"
        cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
        cu_path.write_text(CUDA_SOURCE, encoding="utf-8")
        t0 = time.time()
        try:
            module = load(
                name="lynn_native_p76_cutlass_cute_smoke",
                sources=[str(cpp_path), str(cu_path)],
                build_directory=str(build_root),
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                extra_include_paths=include_paths,
                verbose=args.verbose,
            )
            build_s = time.time() - t0
            out = module.cutlass_cute_smoke()
            torch.cuda.synchronize()
            smoke_values = [float(x) for x in out.cpu().tolist()]
            compile_ok = smoke_values[0] == 4.0 and smoke_values[1] == 8.0 and smoke_values[3] == 120.0
        except Exception as exc:  # noqa: BLE001 - report compile/runtime errors verbatim.
            build_s = time.time() - t0
            error = repr(exc)

    result = {
        "schema_version": "lynn-engine-p76-cutlass-cute-toolchain-probe-v1",
        "torch": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "include_paths": include_paths,
        "required_headers": header_status,
        "missing_headers": missing,
        "compile_attempted": compile_attempted,
        "compile_ok": compile_ok,
        "build_seconds": build_s,
        "smoke_values": smoke_values,
        "error": error,
        "decision": (
            "CUTLASS/CuTe headers and sm_120 FP4 type smoke compile are available."
            if compile_ok
            else "CUTLASS/CuTe toolchain is not ready; do not start the grouped per-16 CuTe kernel until this probe passes."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if compile_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
