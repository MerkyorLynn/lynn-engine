#!/usr/bin/env python3
"""P87: non-uniform FP4 MMA tile layout contract.

P86 proved the missing SM120 E2M1 `<< 2` placement shift restores the all-ones
contract. P87 goes one step closer to a real expert kernel: use the published
CuTe A/B/C register layouts to pack a non-uniform 16x32 by 8x32 logical tile,
run block-scaled FP4 MMA, and compare every lane/register result against a
scalar logical reference.

This still does not use real Lynn expert rows. It proves that our register
packing formulas match CuTe's SM120 fragment layout before we introduce real
per-16 scales and grouped active experts.
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

torch::Tensor p87_layout_tile_contract(int64_t blocks, int64_t scale_byte);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("layout_tile_contract", &p87_layout_tile_contract, "P87 SM120 FP4 layout tile contract");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace {

__device__ __forceinline__ float logical_a(int m, int k) {
  return (((m + 2 * k + 1) % 5) < 2) ? 1.0f : -1.0f;
}

__device__ __forceinline__ float logical_b(int n, int k) {
  return (((3 * n + k + 2) % 7) < 3) ? 1.0f : -1.0f;
}

__device__ __forceinline__ uint8_t e2m1_code_pm1(float value) {
  return value > 0.0f ? 0x2u : 0xAu;
}

__device__ __forceinline__ uint32_t mma_byte_from_e2m1_code(uint8_t code) {
  // CuTe SM120 traits require 0b0000ABCD -> 0b00ABCD00 before MMA.
  return static_cast<uint32_t>((code & 0x0Fu) << 2);
}

__device__ __forceinline__ void fill_a_words(int lane, uint32_t* words) {
  words[0] = 0u;
  words[1] = 0u;
  words[2] = 0u;
  words[3] = 0u;
  const int t0 = lane & 3;
  const int t1 = lane >> 2;
  #pragma unroll
  for (int v = 0; v < 16; ++v) {
    const int v0 = v & 3;
    const int v1 = (v >> 2) & 1;
    const int v2 = (v >> 3) & 1;
    // ALayout = ((4,8),(4,2,2)): ((64,1),(16,8,256))
    const int offset = t0 * 64 + t1 + v0 * 16 + v1 * 8 + v2 * 256;
    const int m = offset & 15;
    const int k = offset >> 4;
    const uint32_t byte = mma_byte_from_e2m1_code(e2m1_code_pm1(logical_a(m, k)));
    words[v >> 2] |= byte << (8 * (v & 3));
  }
}

__device__ __forceinline__ void fill_b_words(int lane, uint32_t* words) {
  words[0] = 0u;
  words[1] = 0u;
  const int t0 = lane & 3;
  const int t1 = lane >> 2;
  #pragma unroll
  for (int v = 0; v < 8; ++v) {
    const int v0 = v & 3;
    const int v1 = (v >> 2) & 1;
    // BLayout = ((4,8),(4,2)): ((32,1),(8,128))
    const int offset = t0 * 32 + t1 + v0 * 8 + v1 * 128;
    const int n = offset & 7;
    const int k = offset >> 3;
    const uint32_t byte = mma_byte_from_e2m1_code(e2m1_code_pm1(logical_b(n, k)));
    words[v >> 2] |= byte << (8 * (v & 3));
  }
}

__device__ __forceinline__ void c_coord_from_lane_value(int lane, int v, int* m, int* n) {
  const int t0 = lane & 3;
  const int t1 = lane >> 2;
  const int v0 = v & 1;
  const int v1 = (v >> 1) & 1;
  // CLayout = ((4,8),(2,2)): ((32,1),(16,8))
  const int offset = t0 * 32 + t1 + v0 * 16 + v1 * 8;
  *m = offset & 15;
  *n = offset >> 4;
}

__device__ __forceinline__ float reference_dot(int m, int n) {
  float acc = 0.0f;
  #pragma unroll
  for (int k = 0; k < 32; ++k) {
    acc += logical_a(m, k) * logical_b(n, k);
  }
  return acc;
}

__global__ void p87_layout_tile_kernel(float* out, uint8_t scale_byte) {
  const int lane = threadIdx.x & 31;
  const int block = blockIdx.x;

  uint32_t a[4];
  uint32_t b[2];
  fill_a_words(lane, a);
  fill_b_words(lane, b);

  float d0 = 0.0f;
  float d1 = 0.0f;
  float d2 = 0.0f;
  float d3 = 0.0f;
  cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
      cute::float_e2m1_t,
      cute::float_e2m1_t,
      float,
      cute::float_ue8m0_t,
      32>::fma(
          d0, d1, d2, d3,
          a[0], a[1], a[2], a[3],
          b[0], b[1],
          0.0f, 0.0f, 0.0f, 0.0f,
          scale_byte, scale_byte);

  const float observed[4] = {d0, d1, d2, d3};
  const int64_t base = (static_cast<int64_t>(block) * 32 + lane) * 4 * 4;
  #pragma unroll
  for (int v = 0; v < 4; ++v) {
    int m = 0;
    int n = 0;
    c_coord_from_lane_value(lane, v, &m, &n);
    const float expected = reference_dot(m, n);
    out[base + v * 4 + 0] = observed[v];
    out[base + v * 4 + 1] = expected;
    out[base + v * 4 + 2] = static_cast<float>(m);
    out[base + v * 4 + 3] = static_cast<float>(n);
  }
}

}  // namespace

torch::Tensor p87_layout_tile_contract(int64_t blocks, int64_t scale_byte) {
  TORCH_CHECK(blocks > 0, "blocks must be positive");
  TORCH_CHECK(scale_byte >= 0 && scale_byte <= 255, "scale_byte must be a byte");
  auto out = torch::empty({blocks, 32, 4, 4}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  p87_layout_tile_kernel<<<static_cast<unsigned int>(blocks), 32>>>(out.data_ptr<float>(), static_cast<uint8_t>(scale_byte));
  const cudaError_t launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess, "p87_layout_tile_kernel launch failed: ", cudaGetErrorString(launch_err));
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
    cpp_path = build_root / "p87_bindings.cpp"
    cu_path = build_root / "p87_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")
    return load(
        name="lynn_native_p87_sm120a_fp4_layout_tile_contract",
        sources=[str(cpp_path), str(cu_path)],
        build_directory=str(build_root),
        extra_cflags=["-O3"],
        extra_cuda_cflags=native_cuda_extra_cuda_cflags(),
        extra_include_paths=discover_native_include_paths(),
        verbose=verbose,
    )


def _time(module, blocks: int, scale_byte: int, repeats: int, warmup: int) -> tuple[torch.Tensor, list[float]]:
    out = None
    for _ in range(warmup):
        out = module.layout_tile_contract(blocks, scale_byte)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = module.layout_tile_contract(blocks, scale_byte)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    assert out is not None
    return out, times_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p87_sm120a_fp4_layout_tile_contract")
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--scale-byte", type=int, default=127)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P87 requires CUDA")

    os.environ.setdefault("LYNN_NATIVE_CUDA_ARCH", "sm_120a")
    _prepare_path()
    t0 = time.time()
    module = _build_module(Path(args.build_dir), args.verbose)
    build_s = time.time() - t0
    out, times_ms = _time(module, args.blocks, args.scale_byte, args.repeats, args.warmup)
    cpu = out.detach().cpu()
    observed = cpu[..., 0]
    expected = cpu[..., 1]
    err = (observed - expected).abs()
    max_idx = int(err.reshape(-1).argmax().item())
    flat = cpu.reshape(-1, 4)
    worst = flat[max_idx]

    result = {
        "schema_version": "lynn-engine-p87-sm120a-fp4-layout-tile-contract-v1",
        "torch": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "include_paths": discover_native_include_paths(),
        "cuda_cflags": native_cuda_extra_cuda_cflags(),
        "build_seconds": build_s,
        "blocks": args.blocks,
        "scale_byte": args.scale_byte,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "times_ms": times_ms,
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "max_abs_err": float(err.max().item()),
        "mean_abs_err": float(err.float().mean().item()),
        "all_exact": bool((err == 0).all().item()),
        "observed_min": float(observed.min().item()),
        "observed_max": float(observed.max().item()),
        "expected_min": float(expected.min().item()),
        "expected_max": float(expected.max().item()),
        "worst": {
            "observed": float(worst[0].item()),
            "expected": float(worst[1].item()),
            "m": int(worst[2].item()),
            "n": int(worst[3].item()),
        },
        "sample_block0_lane0": [
            {
                "observed": float(cpu[0, 0, v, 0].item()),
                "expected": float(cpu[0, 0, v, 1].item()),
                "m": int(cpu[0, 0, v, 2].item()),
                "n": int(cpu[0, 0, v, 3].item()),
            }
            for v in range(4)
        ],
        "decision": (
            "SM120 blockscaled FP4 layout contract PASS: shifted register packing matches scalar logical tile."
            if bool((err == 0).all().item())
            else "SM120 blockscaled FP4 layout contract FAIL: register layout formula still mismatches scalar logical tile."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
