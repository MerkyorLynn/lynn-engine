#!/usr/bin/env python3
"""P86: verify the CuTe FP4 register shift requirement.

CuTe's SM120 MMA traits state that E2M1 values loaded by `ld.matrix b4x16_p64`
must be shifted left by two bits before feeding SM120 F8F6F4/MXF8F6F4 MMA:

  0b0000ABCD -> 0b00ABCD00

P83/P85 manually filled registers with unshifted Lynn nibbles. P86 repeats the
raw and block-scaled all-ones contracts with shifted register words. If the
contract improves, the next active-expert kernel should adopt CuTe's shift step
or its layout/copy helpers rather than treating packed E2M1 bytes as direct MMA
registers.
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

torch::Tensor p86_fp4_shift_contract(int64_t blocks, int64_t scale_byte);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_shift_contract", &p86_fp4_shift_contract, "P86 shifted E2M1 MMA contract");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace {

__device__ __forceinline__ uint32_t shift_fp4_word(uint32_t word) {
  // Apply CuTe's documented SM120 E2M1 placement transform per byte:
  // 0b0000ABCD -> 0b00ABCD00. Mask avoids nibble spill into the next byte.
  return (word << 2) & 0x3C3C3C3Cu;
}

__global__ void p86_shift_kernel(float* out, int64_t blocks, uint8_t scale_byte) {
  const int lane = threadIdx.x & 31;
  const int block = blockIdx.x;

  const uint32_t one_unshifted = 0x22222222u;
  const uint32_t neg_one_unshifted = 0xAAAAAAAAu;
  const uint32_t one_shifted = shift_fp4_word(one_unshifted);
  const uint32_t neg_one_shifted = shift_fp4_word(neg_one_unshifted);

  float rp0 = 0.0f;
  float rp1 = 0.0f;
  float rp2 = 0.0f;
  float rp3 = 0.0f;
  cute::SM120_16x8x32_TN<
      cute::float_e2m1_t,
      cute::float_e2m1_t,
      float>::fma(
          rp0, rp1, rp2, rp3,
          one_shifted, one_shifted, one_shifted, one_shifted,
          one_shifted, one_shifted,
          0.0f, 0.0f, 0.0f, 0.0f);

  float rn0 = 0.0f;
  float rn1 = 0.0f;
  float rn2 = 0.0f;
  float rn3 = 0.0f;
  cute::SM120_16x8x32_TN<
      cute::float_e2m1_t,
      cute::float_e2m1_t,
      float>::fma(
          rn0, rn1, rn2, rn3,
          one_shifted, one_shifted, one_shifted, one_shifted,
          neg_one_shifted, neg_one_shifted,
          0.0f, 0.0f, 0.0f, 0.0f);

  float bp0 = 0.0f;
  float bp1 = 0.0f;
  float bp2 = 0.0f;
  float bp3 = 0.0f;
  cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
      cute::float_e2m1_t,
      cute::float_e2m1_t,
      float,
      cute::float_ue8m0_t,
      32>::fma(
          bp0, bp1, bp2, bp3,
          one_shifted, one_shifted, one_shifted, one_shifted,
          one_shifted, one_shifted,
          0.0f, 0.0f, 0.0f, 0.0f,
          scale_byte, scale_byte);

  float bn0 = 0.0f;
  float bn1 = 0.0f;
  float bn2 = 0.0f;
  float bn3 = 0.0f;
  cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
      cute::float_e2m1_t,
      cute::float_e2m1_t,
      float,
      cute::float_ue8m0_t,
      32>::fma(
          bn0, bn1, bn2, bn3,
          one_shifted, one_shifted, one_shifted, one_shifted,
          neg_one_shifted, neg_one_shifted,
          0.0f, 0.0f, 0.0f, 0.0f,
          scale_byte, scale_byte);

  const int64_t base = (static_cast<int64_t>(block) * 32 + lane) * 8;
  out[base + 0] = rp0;
  out[base + 1] = rp1;
  out[base + 2] = rp2;
  out[base + 3] = rp3;
  out[base + 4] = rn0;
  out[base + 5] = rn1;
  out[base + 6] = rn2;
  out[base + 7] = rn3;

  const int64_t base2 = static_cast<int64_t>(blocks) * 32 * 8 + base;
  out[base2 + 0] = bp0;
  out[base2 + 1] = bp1;
  out[base2 + 2] = bp2;
  out[base2 + 3] = bp3;
  out[base2 + 4] = bn0;
  out[base2 + 5] = bn1;
  out[base2 + 6] = bn2;
  out[base2 + 7] = bn3;
}

}  // namespace

torch::Tensor p86_fp4_shift_contract(int64_t blocks, int64_t scale_byte) {
  TORCH_CHECK(blocks > 0, "blocks must be positive");
  TORCH_CHECK(scale_byte >= 0 && scale_byte <= 255, "scale_byte must be a byte");
  auto out = torch::empty({2, blocks, 32, 8}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  p86_shift_kernel<<<static_cast<unsigned int>(blocks), 32>>>(
      out.data_ptr<float>(), blocks, static_cast<uint8_t>(scale_byte));
  const cudaError_t launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess, "p86_shift_kernel launch failed: ", cudaGetErrorString(launch_err));
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
    cpp_path = build_root / "p86_bindings.cpp"
    cu_path = build_root / "p86_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")
    return load(
        name="lynn_native_p86_sm120a_fp4_shift_contract",
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
        out = module.fp4_shift_contract(blocks, scale_byte)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = module.fp4_shift_contract(blocks, scale_byte)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    assert out is not None
    return out, times_ms


def _summarize(label: str, t: torch.Tensor) -> dict[str, object]:
    pos = t[:, :, :4]
    neg = t[:, :, 4:]
    return {
        "label": label,
        "finite": bool(torch.isfinite(t).all().item()),
        "positive_min": float(pos.min().item()),
        "positive_max": float(pos.max().item()),
        "positive_mean": float(pos.float().mean().item()),
        "negative_min": float(neg.min().item()),
        "negative_max": float(neg.max().item()),
        "negative_mean": float(neg.float().mean().item()),
        "sample_lane0": {
            "positive": [float(x) for x in pos[0, 0, :].tolist()],
            "negative": [float(x) for x in neg[0, 0, :].tolist()],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p86_sm120a_fp4_shift_contract")
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--scale-byte", type=int, default=127)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P86 requires CUDA")

    os.environ.setdefault("LYNN_NATIVE_CUDA_ARCH", "sm_120a")
    _prepare_path()
    t0 = time.time()
    module = _build_module(Path(args.build_dir), args.verbose)
    build_s = time.time() - t0
    out, times_ms = _time(module, args.blocks, args.scale_byte, args.repeats, args.warmup)
    out_cpu = out.detach().cpu()

    raw_summary = _summarize("raw_shifted", out_cpu[0])
    blockscaled_summary = _summarize("blockscaled_shifted", out_cpu[1])
    result = {
        "schema_version": "lynn-engine-p86-sm120a-fp4-shift-contract-v1",
        "torch": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "include_paths": discover_native_include_paths(),
        "cuda_cflags": native_cuda_extra_cuda_cflags(),
        "build_seconds": build_s,
        "blocks": args.blocks,
        "scale_byte": args.scale_byte,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "shift_transform": "(word << 2) & 0x3C3C3C3C",
        "times_ms": times_ms,
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "summaries": [raw_summary, blockscaled_summary],
        "decision": (
            "Shifted E2M1 register fill ran. Compare with P83/P85 to see whether CuTe's required left-shift fixes the all-ones contract."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
