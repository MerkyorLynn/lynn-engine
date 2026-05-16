#!/usr/bin/env python3
"""P85: Blackwell block-scaled FP4 MMA contract probe.

P80-P84 proved that raw `SM120_16x8x32_TN<E2M1,E2M1,F32>` can execute on
R6000 when compiled for `sm_120a`, but the manual operand-register layout does
not satisfy an all-ones dot contract. P85 moves to the official-looking NVFP4
instruction shape in CuTe:

  SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<E2M1,E2M1,F32,UE8M0,32>

This is closer to NVIDIA's Blackwell microscaling/NVFP4 route because the MMA
instruction consumes UE8M0 scale-factor registers. The goal is not yet a full
expert FFN kernel; it is a minimal smoke + all-ones contract that tells us
whether the block-scaled route is the right next branch.
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

torch::Tensor p85_blockscaled_fp4_mma_contract(int64_t blocks, int64_t scale_a, int64_t scale_b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("blockscaled_fp4_mma_contract", &p85_blockscaled_fp4_mma_contract,
        "P85 block-scaled E2M1 MMA contract");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace {

__global__ void p85_blockscaled_kernel(float* out, uint8_t scale_a, uint8_t scale_b) {
  const int lane = threadIdx.x & 31;
  const int block = blockIdx.x;

  const uint32_t zero_word = 0u;
  const uint32_t one_word = 0x22222222u;      // eight +1.0 E2M1 nibbles
  const uint32_t neg_one_word = 0xAAAAAAAAu;  // eight -1.0 E2M1 nibbles

  float c0 = 1.0f;
  float c1 = 2.0f;
  float c2 = 3.0f;
  float c3 = 4.0f;

  // Smoke: zero operands should preserve the accumulator.
  float s0 = c0;
  float s1 = c1;
  float s2 = c2;
  float s3 = c3;
  cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
      cute::float_e2m1_t,
      cute::float_e2m1_t,
      float,
      cute::float_ue8m0_t,
      32>::fma(
          s0, s1, s2, s3,
          zero_word, zero_word, zero_word, zero_word,
          zero_word, zero_word,
          c0, c1, c2, c3,
          scale_a, scale_b);

  // Contract candidate: all +1 over K=32 should act like a scaled +32 dot if
  // the manual register fill matches the instruction fragment contract.
  float p0 = 0.0f;
  float p1 = 0.0f;
  float p2 = 0.0f;
  float p3 = 0.0f;
  cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
      cute::float_e2m1_t,
      cute::float_e2m1_t,
      float,
      cute::float_ue8m0_t,
      32>::fma(
          p0, p1, p2, p3,
          one_word, one_word, one_word, one_word,
          one_word, one_word,
          0.0f, 0.0f, 0.0f, 0.0f,
          scale_a, scale_b);

  float n0 = 0.0f;
  float n1 = 0.0f;
  float n2 = 0.0f;
  float n3 = 0.0f;
  cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
      cute::float_e2m1_t,
      cute::float_e2m1_t,
      float,
      cute::float_ue8m0_t,
      32>::fma(
          n0, n1, n2, n3,
          one_word, one_word, one_word, one_word,
          neg_one_word, neg_one_word,
          0.0f, 0.0f, 0.0f, 0.0f,
          scale_a, scale_b);

  const int64_t base = (static_cast<int64_t>(block) * 32 + lane) * 12;
  out[base + 0] = s0;
  out[base + 1] = s1;
  out[base + 2] = s2;
  out[base + 3] = s3;
  out[base + 4] = p0;
  out[base + 5] = p1;
  out[base + 6] = p2;
  out[base + 7] = p3;
  out[base + 8] = n0;
  out[base + 9] = n1;
  out[base + 10] = n2;
  out[base + 11] = n3;
}

}  // namespace

torch::Tensor p85_blockscaled_fp4_mma_contract(int64_t blocks, int64_t scale_a, int64_t scale_b) {
  TORCH_CHECK(blocks > 0, "blocks must be positive");
  TORCH_CHECK(scale_a >= 0 && scale_a <= 255, "scale_a must be a byte");
  TORCH_CHECK(scale_b >= 0 && scale_b <= 255, "scale_b must be a byte");
  auto out = torch::empty({blocks, 32, 12}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  p85_blockscaled_kernel<<<static_cast<unsigned int>(blocks), 32>>>(
      out.data_ptr<float>(), static_cast<uint8_t>(scale_a), static_cast<uint8_t>(scale_b));
  const cudaError_t launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess, "p85_blockscaled_kernel launch failed: ", cudaGetErrorString(launch_err));
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
    cpp_path = build_root / "p85_bindings.cpp"
    cu_path = build_root / "p85_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")
    return load(
        name="lynn_native_p85_sm120a_blockscaled_fp4_mma_contract",
        sources=[str(cpp_path), str(cu_path)],
        build_directory=str(build_root),
        extra_cflags=["-O3"],
        extra_cuda_cflags=native_cuda_extra_cuda_cflags(),
        extra_include_paths=discover_native_include_paths(),
        verbose=verbose,
    )


def _run_once(module, blocks: int, scale_a: int, scale_b: int) -> torch.Tensor:
    out = module.blockscaled_fp4_mma_contract(blocks, scale_a, scale_b)
    torch.cuda.synchronize()
    return out


def _time(module, blocks: int, scale_a: int, scale_b: int, repeats: int, warmup: int) -> tuple[torch.Tensor, list[float]]:
    out = None
    for _ in range(warmup):
        out = _run_once(module, blocks, scale_a, scale_b)

    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = module.blockscaled_fp4_mma_contract(blocks, scale_a, scale_b)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    assert out is not None
    return out, times_ms


def _summarize_tensor(out: torch.Tensor) -> dict[str, object]:
    out_cpu = out.detach().cpu()
    smoke = out_cpu[:, :, :4]
    pos = out_cpu[:, :, 4:8]
    neg = out_cpu[:, :, 8:12]
    smoke_expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 4)
    smoke_err = (smoke - smoke_expected).abs()

    return {
        "finite": bool(torch.isfinite(out_cpu).all().item()),
        "smoke_passthrough_max_abs_err": float(smoke_err.max().item()),
        "smoke_sample_lane0": [float(x) for x in smoke[0, 0, :].tolist()],
        "positive_min": float(pos.min().item()),
        "positive_max": float(pos.max().item()),
        "positive_mean": float(pos.float().mean().item()),
        "positive_nonzero_count": int((pos != 0).sum().item()),
        "negative_min": float(neg.min().item()),
        "negative_max": float(neg.max().item()),
        "negative_mean": float(neg.float().mean().item()),
        "negative_nonzero_count": int((neg != 0).sum().item()),
        "sample_lane0": {
            "positive": [float(x) for x in pos[0, 0, :].tolist()],
            "negative": [float(x) for x in neg[0, 0, :].tolist()],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p85_sm120a_blockscaled_fp4_mma_contract")
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--scale-bytes", default="126,127,128")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P85 requires CUDA")

    os.environ.setdefault("LYNN_NATIVE_CUDA_ARCH", "sm_120a")
    _prepare_path()

    t0 = time.time()
    module = _build_module(Path(args.build_dir), args.verbose)
    build_s = time.time() - t0

    scale_bytes = [int(x.strip()) for x in args.scale_bytes.split(",") if x.strip()]
    rows: list[dict[str, object]] = []
    for byte in scale_bytes:
        out, times_ms = _time(module, args.blocks, byte, byte, args.repeats, args.warmup)
        summary = _summarize_tensor(out)
        rows.append({
            "scale_a": byte,
            "scale_b": byte,
            "times_ms": times_ms,
            "mean_ms": statistics.fmean(times_ms),
            "median_ms": statistics.median(times_ms),
            "min_ms": min(times_ms),
            "max_ms": max(times_ms),
            **summary,
        })

    result = {
        "schema_version": "lynn-engine-p85-sm120a-blockscaled-fp4-mma-contract-v1",
        "torch": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "include_paths": discover_native_include_paths(),
        "cuda_cflags": native_cuda_extra_cuda_cflags(),
        "build_seconds": build_s,
        "blocks": args.blocks,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "scale_bytes": scale_bytes,
        "rows": rows,
        "zero_operand_smoke_ok": all(row["finite"] and row["smoke_passthrough_max_abs_err"] == 0.0 for row in rows),
        "decision": (
            "BLOCKSCALED FP4 MMA compile/runtime smoke PASS. Inspect all-ones contract rows to decide whether manual register layout is usable."
            if rows and all(row["finite"] for row in rows)
            else "BLOCKSCALED FP4 MMA did not produce finite runtime output."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
