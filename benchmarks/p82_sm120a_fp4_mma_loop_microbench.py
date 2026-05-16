#!/usr/bin/env python3
"""P82: raw sm_120a E2M1 MMA loop microbench.

P80 proved a single CuTe E2M1 MMA instruction can execute through a forced
`sm_120a` torch extension. P82 turns that into a timing probe with non-zero
E2M1 operands and the shared Lynn native CUDA arch policy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.native_cuda import (  # noqa: E402
    discover_native_include_paths,
    native_cuda_extra_cuda_cflags,
)


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor p82_fp4_mma_loop(int64_t blocks, int64_t iters);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_mma_loop", &p82_fp4_mma_loop, "P82 raw sm120a E2M1 MMA loop");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace {

__global__ void p82_mma_loop_kernel(float* out, int iters) {
  const int lane = threadIdx.x & 31;
  const int block = blockIdx.x;

  float d0 = 1.0f + static_cast<float>(lane & 3);
  float d1 = 2.0f + static_cast<float>(lane & 3);
  float d2 = 3.0f + static_cast<float>(lane & 3);
  float d3 = 4.0f + static_cast<float>(lane & 3);

  // E2M1 nibble 0x2 encodes +1.0 and 0xA encodes -1.0. Use non-zero patterns
  // so this is not merely the all-zero smoke from P80.
  uint32_t a0 = 0x22222222u ^ static_cast<uint32_t>(lane);
  uint32_t a1 = 0x22222222u;
  uint32_t a2 = 0x22222222u;
  uint32_t a3 = 0x22222222u;
  uint32_t b0 = 0x22222222u;
  uint32_t b1 = 0x2A222222u;

  for (int i = 0; i < iters; ++i) {
    cute::SM120_16x8x32_TN<
        cute::float_e2m1_t,
        cute::float_e2m1_t,
        float>::fma(
            d0, d1, d2, d3,
            a0, a1, a2, a3,
            b0, b1,
            d0, d1, d2, d3);
  }

  const int64_t offset = (static_cast<int64_t>(block) * 32 + lane) * 4;
  out[offset + 0] = d0;
  out[offset + 1] = d1;
  out[offset + 2] = d2;
  out[offset + 3] = d3;
}

}  // namespace

torch::Tensor p82_fp4_mma_loop(int64_t blocks, int64_t iters) {
  TORCH_CHECK(blocks > 0, "blocks must be positive");
  TORCH_CHECK(iters > 0, "iters must be positive");
  auto out = torch::empty({blocks, 32, 4}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  p82_mma_loop_kernel<<<static_cast<unsigned int>(blocks), 32>>>(out.data_ptr<float>(), static_cast<int>(iters));
  const cudaError_t launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess, "p82_mma_loop_kernel launch failed: ", cudaGetErrorString(launch_err));
  return out;
}
"""


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


def _build_module(build_root: Path, verbose: bool):
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True, exist_ok=True)
    cpp_path = build_root / "p82_bindings.cpp"
    cu_path = build_root / "p82_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")
    return load(
        name="lynn_native_p82_sm120a_fp4_mma_loop",
        sources=[str(cpp_path), str(cu_path)],
        build_directory=str(build_root),
        extra_cflags=["-O3"],
        extra_cuda_cflags=native_cuda_extra_cuda_cflags(),
        extra_include_paths=discover_native_include_paths(),
        verbose=verbose,
    )


def _time_kernel(module, blocks: int, iters: int, repeats: int, warmup: int) -> dict[str, object]:
    for _ in range(warmup):
        out = module.fp4_mma_loop(blocks, iters)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    out = None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = module.fp4_mma_loop(blocks, iters)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))

    assert out is not None
    out_cpu = out.detach().cpu()
    total_mma = blocks * iters
    mean_ms = statistics.fmean(times_ms)
    return {
        "blocks": blocks,
        "iters": iters,
        "repeats": repeats,
        "warmup": warmup,
        "times_ms": times_ms,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "warp_mma_instructions": total_mma,
        "warp_mma_per_ms": total_mma / mean_ms,
        "output_sample": [float(x) for x in out_cpu.reshape(-1)[:16].tolist()],
        "output_sum": float(out_cpu.sum().item()),
        "output_all_finite": bool(torch.isfinite(out_cpu).all().item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p82_sm120a_fp4_mma_loop")
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P82 requires CUDA")

    # P80 proved sm_120a is the required feature target for the R6000 FP4 MMA
    # instruction path. Use the shared native policy rather than hardcoding
    # benchmark-local flags.
    os.environ.setdefault("LYNN_NATIVE_CUDA_ARCH", "sm_120a")
    _prepare_path()

    t0 = time.time()
    module = _build_module(Path(args.build_dir), args.verbose)
    build_s = time.time() - t0
    timing = _time_kernel(module, args.blocks, args.iters, args.repeats, args.warmup)

    ok = bool(timing["output_all_finite"]) and timing["warp_mma_per_ms"] > 0
    result = {
        "schema_version": "lynn-engine-p82-sm120a-fp4-mma-loop-microbench-v1",
        "torch": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "include_paths": discover_native_include_paths(),
        "cuda_cflags": native_cuda_extra_cuda_cflags(),
        "build_seconds": build_s,
        "timing": timing,
        "ok": ok,
        "decision": (
            "Raw sm_120a E2M1 MMA loop runs under the shared Lynn native CUDA policy."
            if ok
            else "Raw sm_120a E2M1 MMA loop failed; inspect build/runtime output before adding scale/weight plumbing."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
