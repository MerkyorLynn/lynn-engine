#!/usr/bin/env python3
"""P88: real Lynn gate/up packed tile through SM120 block-scaled FP4 MMA.

P87 proved the synthetic tile layout. P88 feeds real Lynn packed gate/up rows
and activation codes into that same layout for one 16x32 K tile and compares
against a scalar code-level reference.

This intentionally tests the packed E2M1 codes first, with neutral UE8M0 scale.
Real Lynn per-16 floating scales are a separate P89 problem. Keeping P88
code-level makes failures diagnosable: if this passes, the remaining challenge
is scale adaptation, not fragment layout or real tensor indexing.
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p10e_packed_active_expert_probe import (  # noqa: E402
    _load_grouped,
    _prefill_to_layer_input,
)
from engine.full_forward import _rms_norm  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_linear import quantize_fp4_m1_native  # noqa: E402
from engine.native_cuda import (  # noqa: E402
    discover_native_include_paths,
    native_cuda_extra_cuda_cflags,
)


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor p88_real_gateup_tile_contract(
    torch::Tensor act_packed,
    torch::Tensor weight_packed,
    int64_t row_offset,
    int64_t k_offset,
    int64_t scale_byte);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("real_gateup_tile_contract", &p88_real_gateup_tile_contract,
        "P88 real Lynn gate/up packed tile contract");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace {

__device__ __forceinline__ float e2m1_from_code(uint8_t code) {
  const uint8_t mag = code & 0x07u;
  const bool sign = (code & 0x08u) != 0;
  float v = 0.0f;
  if (mag == 1) {
    v = 0.5f;
  } else if (mag == 2) {
    v = 1.0f;
  } else if (mag == 3) {
    v = 1.5f;
  } else if (mag == 4) {
    v = 2.0f;
  } else if (mag == 5) {
    v = 3.0f;
  } else if (mag == 6) {
    v = 4.0f;
  } else if (mag == 7) {
    v = 6.0f;
  }
  return sign ? -v : v;
}

__device__ __forceinline__ uint8_t get_code_from_packed(const uint8_t* ptr, int elem) {
  const uint8_t byte = ptr[elem >> 1];
  return (elem & 1) == 0 ? (byte & 0x0Fu) : ((byte >> 4) & 0x0Fu);
}

__device__ __forceinline__ uint32_t mma_byte_from_code(uint8_t code) {
  return static_cast<uint32_t>((code & 0x0Fu) << 2);
}

__device__ __forceinline__ void fill_a_words(const uint8_t* act, int k_offset, int lane, uint32_t* words) {
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
    const int offset = t0 * 64 + t1 + v0 * 16 + v1 * 8 + v2 * 256;
    const int k = offset >> 4;
    const uint8_t code = get_code_from_packed(act, k_offset + k);
    words[v >> 2] |= mma_byte_from_code(code) << (8 * (v & 3));
  }
}

__device__ __forceinline__ void fill_b_words(
    const uint8_t* weight, int row_offset, int k_offset, int packed_stride_m, int lane, uint32_t* words) {
  words[0] = 0u;
  words[1] = 0u;
  const int t0 = lane & 3;
  const int t1 = lane >> 2;
  #pragma unroll
  for (int v = 0; v < 8; ++v) {
    const int v0 = v & 3;
    const int v1 = (v >> 2) & 1;
    const int offset = t0 * 32 + t1 + v0 * 8 + v1 * 128;
    const int n = offset & 7;
    const int k = offset >> 3;
    const uint8_t code = get_code_from_packed(weight + (row_offset + n) * packed_stride_m, k_offset + k);
    words[v >> 2] |= mma_byte_from_code(code) << (8 * (v & 3));
  }
}

__device__ __forceinline__ void c_coord_from_lane_value(int lane, int v, int* m, int* n) {
  const int t0 = lane & 3;
  const int t1 = lane >> 2;
  const int v0 = v & 1;
  const int v1 = (v >> 1) & 1;
  const int offset = t0 * 32 + t1 + v0 * 16 + v1 * 8;
  *m = offset & 15;
  *n = offset >> 4;
}

__device__ __forceinline__ float scalar_reference(
    const uint8_t* act, const uint8_t* weight, int row_offset, int k_offset, int packed_stride_m, int m, int n) {
  (void)m;  // P88 uses one activation row broadcast across the 16 MMA M rows.
  float acc = 0.0f;
  #pragma unroll
  for (int k = 0; k < 32; ++k) {
    const uint8_t a_code = get_code_from_packed(act, k_offset + k);
    const uint8_t b_code = get_code_from_packed(weight + (row_offset + n) * packed_stride_m, k_offset + k);
    acc += e2m1_from_code(a_code) * e2m1_from_code(b_code);
  }
  return acc;
}

__global__ void p88_real_tile_kernel(
    const uint8_t* __restrict__ act,
    const uint8_t* __restrict__ weight,
    float* __restrict__ out,
    int row_offset,
    int k_offset,
    int packed_stride_m,
    uint8_t scale_byte) {
  const int lane = threadIdx.x & 31;
  const int block = blockIdx.x;

  uint32_t a[4];
  uint32_t b[2];
  fill_a_words(act, k_offset, lane, a);
  fill_b_words(weight, row_offset, k_offset, packed_stride_m, lane, b);

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
    const float expected = scalar_reference(act, weight, row_offset, k_offset, packed_stride_m, m, n);
    out[base + v * 4 + 0] = observed[v];
    out[base + v * 4 + 1] = expected;
    out[base + v * 4 + 2] = static_cast<float>(m);
    out[base + v * 4 + 3] = static_cast<float>(n);
  }
}

}  // namespace

torch::Tensor p88_real_gateup_tile_contract(
    torch::Tensor act_packed,
    torch::Tensor weight_packed,
    int64_t row_offset,
    int64_t k_offset,
    int64_t scale_byte) {
  TORCH_CHECK(act_packed.is_cuda(), "act_packed must be CUDA");
  TORCH_CHECK(weight_packed.is_cuda(), "weight_packed must be CUDA");
  TORCH_CHECK(act_packed.scalar_type() == torch::kUInt8, "act_packed must be uint8");
  TORCH_CHECK(weight_packed.scalar_type() == torch::kUInt8, "weight_packed must be uint8");
  TORCH_CHECK(act_packed.dim() == 1, "act_packed must be [K/2]");
  TORCH_CHECK(weight_packed.dim() == 2, "weight_packed must be [rows,K/2]");
  TORCH_CHECK(row_offset >= 0 && row_offset + 8 <= weight_packed.size(0), "row_offset out of bounds");
  TORCH_CHECK(k_offset >= 0 && k_offset + 32 <= act_packed.size(0) * 2, "k_offset out of bounds");
  TORCH_CHECK((k_offset % 2) == 0, "k_offset must be even for packed-byte addressing");
  TORCH_CHECK(scale_byte >= 0 && scale_byte <= 255, "scale_byte must be a byte");
  auto act_c = act_packed.contiguous();
  auto weight_c = weight_packed.contiguous();
  auto out = torch::empty({1, 32, 4, 4}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  p88_real_tile_kernel<<<1, 32>>>(
      act_c.data_ptr<uint8_t>(),
      weight_c.data_ptr<uint8_t>(),
      out.data_ptr<float>(),
      static_cast<int>(row_offset),
      static_cast<int>(k_offset),
      static_cast<int>(weight_c.stride(0)),
      static_cast<uint8_t>(scale_byte));
  const cudaError_t launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess, "p88_real_tile_kernel launch failed: ", cudaGetErrorString(launch_err));
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
    cpp_path = build_root / "p88_bindings.cpp"
    cu_path = build_root / "p88_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")
    return load(
        name="lynn_native_p88_sm120a_real_gateup_tile_contract",
        sources=[str(cpp_path), str(cu_path)],
        build_directory=str(build_root),
        extra_cflags=["-O3"],
        extra_cuda_cflags=native_cuda_extra_cuda_cflags(),
        extra_include_paths=discover_native_include_paths(),
        verbose=verbose,
    )


def _time(fn, repeats: int, warmup: int) -> tuple[torch.Tensor, list[float]]:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    assert out is not None
    return out, times_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p88_sm120a_real_gateup_tile_contract")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--expert-slot", type=int, default=0)
    ap.add_argument("--row-offset", type=int, default=0)
    ap.add_argument("--k-offset", type=int, default=0)
    ap.add_argument("--scale-byte", type=int, default=127)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P88 requires CUDA")
    os.environ.setdefault("LYNN_NATIVE_CUDA_ARCH", "sm_120a")
    _prepare_path()

    model_dir = Path(args.model)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    h_layer, _ = _prefill_to_layer_input(runner, args.layer, args.prompt)
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    hidden_2d = h_moe.view(-1, h_moe.shape[-1])[:1].contiguous()

    router_logits = F.linear(hidden_2d, w["mlp.gate.weight"])
    _, expert_indices = torch.topk(router_logits, int(cfg["num_experts_per_tok"]), dim=-1)
    expert_ids = expert_indices[0].to(torch.long)
    if not (0 <= args.expert_slot < int(expert_ids.numel())):
        raise ValueError(f"expert-slot must be in [0,{int(expert_ids.numel())})")
    expert_id = int(expert_ids[args.expert_slot].item())

    gate_up_packed, gate_up_scale, gate_up_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{args.layer}.mlp.experts.gate_up_proj",
        runner.device,
    )
    act_packed, act_scale = quantize_fp4_m1_native(hidden_2d)
    act_packed_1d = act_packed[0].contiguous()
    weight_packed = gate_up_packed[expert_id].contiguous()

    if args.row_offset + 8 > int(weight_packed.shape[0]):
        raise ValueError("row-offset + 8 exceeds gate/up rows")
    if args.k_offset + 32 > int(hidden_2d.shape[-1]):
        raise ValueError("k-offset + 32 exceeds hidden size")
    if args.k_offset % 2:
        raise ValueError("k-offset must be even")

    t0 = time.time()
    module = _build_module(Path(args.build_dir), args.verbose)
    build_s = time.time() - t0

    def run_tile() -> torch.Tensor:
        return module.real_gateup_tile_contract(
            act_packed_1d,
            weight_packed,
            args.row_offset,
            args.k_offset,
            args.scale_byte,
        )

    out, times_ms = _time(run_tile, args.repeats, args.warmup)
    cpu = out.detach().cpu()
    observed = cpu[..., 0]
    expected = cpu[..., 1]
    err = (observed - expected).abs()
    max_idx = int(err.reshape(-1).argmax().item())
    flat = cpu.reshape(-1, 4)
    worst = flat[max_idx]

    scale_groups = {
        "activation_native_scale_sample": [float(x) for x in act_scale[:8].detach().float().cpu().tolist()],
        "weight_rows": [
            [float(x) for x in gate_up_scale[expert_id, args.row_offset + r, args.k_offset // 16 : args.k_offset // 16 + 2].detach().cpu().tolist()]
            for r in range(8)
        ],
        "weight_global": float(gate_up_global.detach().float().cpu().reshape(-1)[0].item()),
    }

    result = {
        "schema_version": "lynn-engine-p88-sm120a-real-gateup-tile-contract-v1",
        "model": args.model,
        "layer": args.layer,
        "expert_slot": args.expert_slot,
        "expert_id": expert_id,
        "row_offset": args.row_offset,
        "k_offset": args.k_offset,
        "scale_byte": args.scale_byte,
        "top_k_expert_ids": [int(x) for x in expert_ids.tolist()],
        "torch": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "include_paths": discover_native_include_paths(),
        "cuda_cflags": native_cuda_extra_cuda_cflags(),
        "build_seconds": build_s,
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
        "scale_groups_not_applied_in_p88": scale_groups,
        "decision": (
            "Real Lynn gate/up packed-code tile PASS: SM120 shifted blockscaled MMA matches scalar code-level dot."
            if bool((err == 0).all().item())
            else "Real Lynn gate/up packed-code tile FAIL: inspect indexing/layout before adding real per-16 scales."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
