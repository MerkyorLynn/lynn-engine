#!/usr/bin/env python3
"""P83: all-ones E2M1 MMA contract for Lynn packed nibbles.

P82 proved a raw sm_120a FP4 MMA loop can execute. P83 uses the Lynn/CUTLASS
compatible E2M1 nibble code for +1.0 (0x2) and -1.0 (0xA) and checks the basic
warp-level math contract:

  all(+1) A x all(+1) B over K=32 -> +32
  all(+1) A x all(-1) B over K=32 -> -32

This is intentionally simpler than the final active expert layout. It verifies
that direct Lynn nibble bytes can populate MMA registers and produce the scalar
reference result before per-16 scales and real expert rows enter the picture.
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

torch::Tensor p83_fp4_mma_allones(int64_t blocks);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_mma_allones", &p83_fp4_mma_allones, "P83 all-ones E2M1 MMA contract");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace {

__global__ void p83_allones_kernel(float* out) {
  const int lane = threadIdx.x & 31;
  const int block = blockIdx.x;

  const uint32_t one_word = 0x22222222u;  // eight +1.0 E2M1 nibbles
  const uint32_t neg_one_word = 0xAAAAAAAAu;  // eight -1.0 E2M1 nibbles

  const uint32_t a0 = one_word;
  const uint32_t a1 = one_word;
  const uint32_t a2 = one_word;
  const uint32_t a3 = one_word;
  const uint32_t b_pos0 = one_word;
  const uint32_t b_pos1 = one_word;
  const uint32_t b_neg0 = neg_one_word;
  const uint32_t b_neg1 = neg_one_word;

  float p0 = 0.0f;
  float p1 = 0.0f;
  float p2 = 0.0f;
  float p3 = 0.0f;
  cute::SM120_16x8x32_TN<
      cute::float_e2m1_t,
      cute::float_e2m1_t,
      float>::fma(
          p0, p1, p2, p3,
          a0, a1, a2, a3,
          b_pos0, b_pos1,
          0.0f, 0.0f, 0.0f, 0.0f);

  float n0 = 0.0f;
  float n1 = 0.0f;
  float n2 = 0.0f;
  float n3 = 0.0f;
  cute::SM120_16x8x32_TN<
      cute::float_e2m1_t,
      cute::float_e2m1_t,
      float>::fma(
          n0, n1, n2, n3,
          a0, a1, a2, a3,
          b_neg0, b_neg1,
          0.0f, 0.0f, 0.0f, 0.0f);

  const int64_t base = (static_cast<int64_t>(block) * 32 + lane) * 8;
  out[base + 0] = p0;
  out[base + 1] = p1;
  out[base + 2] = p2;
  out[base + 3] = p3;
  out[base + 4] = n0;
  out[base + 5] = n1;
  out[base + 6] = n2;
  out[base + 7] = n3;
}

}  // namespace

torch::Tensor p83_fp4_mma_allones(int64_t blocks) {
  TORCH_CHECK(blocks > 0, "blocks must be positive");
  auto out = torch::empty({blocks, 32, 8}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  p83_allones_kernel<<<static_cast<unsigned int>(blocks), 32>>>(out.data_ptr<float>());
  const cudaError_t launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess, "p83_allones_kernel launch failed: ", cudaGetErrorString(launch_err));
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
    cpp_path = build_root / "p83_bindings.cpp"
    cu_path = build_root / "p83_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")
    return load(
        name="lynn_native_p83_sm120a_fp4_mma_allones",
        sources=[str(cpp_path), str(cu_path)],
        build_directory=str(build_root),
        extra_cflags=["-O3"],
        extra_cuda_cflags=native_cuda_extra_cuda_cflags(),
        extra_include_paths=discover_native_include_paths(),
        verbose=verbose,
    )


def _time(module, blocks: int, repeats: int, warmup: int) -> tuple[torch.Tensor, list[float]]:
    out = None
    for _ in range(warmup):
        out = module.fp4_mma_allones(blocks)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = module.fp4_mma_allones(blocks)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    assert out is not None
    return out, times_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p83_sm120a_fp4_mma_allones")
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P83 requires CUDA")

    os.environ.setdefault("LYNN_NATIVE_CUDA_ARCH", "sm_120a")
    _prepare_path()

    t0 = time.time()
    module = _build_module(Path(args.build_dir), args.verbose)
    build_s = time.time() - t0
    out, times_ms = _time(module, args.blocks, args.repeats, args.warmup)
    out_cpu = out.detach().cpu()

    pos = out_cpu[:, :, :4]
    neg = out_cpu[:, :, 4:]
    pos_err = (pos - 32.0).abs()
    neg_err = (neg + 32.0).abs()
    max_abs_err = max(float(pos_err.max().item()), float(neg_err.max().item()))
    ok = bool(torch.isfinite(out_cpu).all().item()) and max_abs_err == 0.0

    result = {
        "schema_version": "lynn-engine-p83-sm120a-fp4-mma-allones-contract-v1",
        "torch": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "include_paths": discover_native_include_paths(),
        "cuda_cflags": native_cuda_extra_cuda_cflags(),
        "build_seconds": build_s,
        "blocks": args.blocks,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "times_ms": times_ms,
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "positive_expected": 32.0,
        "negative_expected": -32.0,
        "positive_min": float(pos.min().item()),
        "positive_max": float(pos.max().item()),
        "negative_min": float(neg.min().item()),
        "negative_max": float(neg.max().item()),
        "max_abs_err": max_abs_err,
        "output_sample": [float(x) for x in out_cpu.reshape(-1)[:24].tolist()],
        "ok": ok,
        "decision": (
            "All-ones Lynn/CUTLASS E2M1 nibble contract matches scalar K=32 reference exactly."
            if ok
            else "All-ones E2M1 MMA contract mismatch; inspect register or nibble layout before real packed expert rows."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
