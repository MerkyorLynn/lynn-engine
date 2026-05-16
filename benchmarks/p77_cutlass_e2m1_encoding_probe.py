#!/usr/bin/env python3
"""P77: compare Lynn E2M1 nibble table with CUTLASS float_e2m1_t bitcast.

The grouped CuTe/CUTLASS kernel can only consume the current packed bytes
directly if Lynn's nibble encoding is storage-compatible with
cutlass::float_e2m1_t. If not, the P77 result defines the required remap.
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


LYNN_E2M1_TABLE = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor p77_cutlass_e2m1_table();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cutlass_e2m1_table", &p77_cutlass_e2m1_table, "P77 CUTLASS E2M1 bitcast table");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cutlass/float_subbyte.h>

namespace {

__global__ void p77_table_kernel(float* out) {
  const int i = threadIdx.x;
  if (i < 16) {
    auto value = cutlass::float_e2m1_t::bitcast(static_cast<unsigned char>(i));
    out[i] = static_cast<float>(value);
  }
}

}  // namespace

torch::Tensor p77_cutlass_e2m1_table() {
  auto out = torch::empty({16}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  p77_table_kernel<<<1, 32>>>(out.data_ptr<float>());
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "p77_table_kernel launch failed: ", cudaGetErrorString(err));
  return out;
}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p77_cutlass_e2m1_encoding")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P77 requires CUDA")

    python_bin = Path(sys.executable).resolve().parent
    os.environ["PATH"] = f"{python_bin}:{os.environ.get('PATH', '')}"
    for extra_bin in (Path.home() / "miniconda3" / "bin", Path("/root/miniconda3/bin")):
        if extra_bin.exists():
            os.environ["PATH"] = f"{extra_bin}:{os.environ.get('PATH', '')}"

    include_paths = discover_native_include_paths()
    build_root = Path(args.build_dir)
    build_root.mkdir(parents=True, exist_ok=True)
    cpp_path = build_root / "p77_cutlass_e2m1_bindings.cpp"
    cu_path = build_root / "p77_cutlass_e2m1_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")

    t0 = time.time()
    module = load(
        name="lynn_native_p77_cutlass_e2m1_encoding",
        sources=[str(cpp_path), str(cu_path)],
        build_directory=str(build_root),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_include_paths=include_paths,
        verbose=args.verbose,
    )
    build_s = time.time() - t0
    table = module.cutlass_e2m1_table()
    torch.cuda.synchronize()
    cutlass_table = [float(x) for x in table.cpu().tolist()]
    diffs = [cutlass_table[i] - LYNN_E2M1_TABLE[i] for i in range(16)]
    exact_storage_compatible = all(abs(x) == 0 for x in diffs)

    remap_lynn_to_cutlass = {}
    for lynn_idx, lynn_val in enumerate(LYNN_E2M1_TABLE):
        matches = [idx for idx, val in enumerate(cutlass_table) if val == lynn_val]
        remap_lynn_to_cutlass[str(lynn_idx)] = matches[0] if matches else None

    result = {
        "schema_version": "lynn-engine-p77-cutlass-e2m1-encoding-probe-v1",
        "include_paths": include_paths,
        "build_seconds": build_s,
        "lynn_table": LYNN_E2M1_TABLE,
        "cutlass_bitcast_table": cutlass_table,
        "diffs_cutlass_minus_lynn": diffs,
        "exact_storage_compatible": exact_storage_compatible,
        "remap_lynn_nibble_to_cutlass_nibble": remap_lynn_to_cutlass,
        "decision": (
            "Lynn packed E2M1 nibble bytes are CUTLASS storage-compatible."
            if exact_storage_compatible
            else "Lynn E2M1 values are not storage-compatible with CUTLASS bitcast order; grouped kernel needs a remap/repack layer or a custom decode path."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if exact_storage_compatible else 2


if __name__ == "__main__":
    raise SystemExit(main())
