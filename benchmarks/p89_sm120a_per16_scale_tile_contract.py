#!/usr/bin/env python3
"""P89: real Lynn per-16 scale strategy for SM120 FP4 MMA tiles.

P88 proved real Lynn packed activation/weight codes can enter SM120
block-scaled FP4 MMA exactly. P89 adds Lynn's current per-16 scale contract.

The SM120 MXF8F6F4 K=32 instruction has one scale vector per 32 elements, while
Lynn's artifact stores a separate floating scale per 16 elements. P89 therefore
tests the exact current-artifact strategy:

  1. run one MMA for K[0:16] with K[16:32] zeroed;
  2. run one MMA for K[16:32] with K[0:16] zeroed;
  3. apply the two real per-16 scale products and sum.

It also quantifies how wrong a single folded K32 scale would be for the same
tile. This decides whether the current Lynn-native artifact can be consumed
exactly, and what we lose if we try to force it into one scale per K32.
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
from engine.native_cuda import (  # noqa: E402
    discover_native_include_paths,
    native_cuda_extra_cuda_cflags,
)
from engine.nvfp4_runtime import _quantize_activation_to_fp4  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor p89_per16_scale_tile_contract(
    torch::Tensor act_packed,
    torch::Tensor act_scale,
    torch::Tensor weight_packed,
    torch::Tensor weight_scale,
    torch::Tensor weight_global_scale,
    int64_t row_offset,
    int64_t k_offset,
    int64_t scale_byte);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("per16_scale_tile_contract", &p89_per16_scale_tile_contract,
        "P89 real Lynn per-16 scale tile contract");
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

__device__ __forceinline__ void fill_a_words_group(
    const uint8_t* act, int k_offset, int group16, int lane, uint32_t* words) {
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
    uint8_t code = 0u;
    if ((k >> 4) == group16) {
      code = get_code_from_packed(act, k_offset + k);
    }
    words[v >> 2] |= mma_byte_from_code(code) << (8 * (v & 3));
  }
}

__device__ __forceinline__ void fill_b_words_group(
    const uint8_t* weight, int row_offset, int k_offset, int packed_stride_m, int group16, int lane, uint32_t* words) {
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
    uint8_t code = 0u;
    if ((k >> 4) == group16) {
      code = get_code_from_packed(weight + (row_offset + n) * packed_stride_m, k_offset + k);
    }
    words[v >> 2] |= mma_byte_from_code(code) << (8 * (v & 3));
  }
}

__device__ __forceinline__ void run_mma_group(
    const uint8_t* act,
    const uint8_t* weight,
    int row_offset,
    int k_offset,
    int packed_stride_m,
    int group16,
    int lane,
    uint8_t scale_byte,
    float* d) {
  uint32_t a[4];
  uint32_t b[2];
  fill_a_words_group(act, k_offset, group16, lane, a);
  fill_b_words_group(weight, row_offset, k_offset, packed_stride_m, group16, lane, b);
  d[0] = 0.0f;
  d[1] = 0.0f;
  d[2] = 0.0f;
  d[3] = 0.0f;
  cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
      cute::float_e2m1_t,
      cute::float_e2m1_t,
      float,
      cute::float_ue8m0_t,
      32>::fma(
          d[0], d[1], d[2], d[3],
          a[0], a[1], a[2], a[3],
          b[0], b[1],
          0.0f, 0.0f, 0.0f, 0.0f,
          scale_byte, scale_byte);
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

__device__ __forceinline__ float scalar_scaled_reference(
    const uint8_t* act,
    const float* act_scale,
    const uint8_t* weight,
    const float* weight_scale,
    float weight_global,
    int row_offset,
    int k_offset,
    int packed_stride_m,
    int scale_stride_m,
    int scale_stride_g,
    int n) {
  float acc = 0.0f;
  #pragma unroll
  for (int k = 0; k < 32; ++k) {
    const int global_k = k_offset + k;
    const int g = global_k >> 4;
    const uint8_t a_code = get_code_from_packed(act, global_k);
    const uint8_t b_code = get_code_from_packed(weight + (row_offset + n) * packed_stride_m, global_k);
    const float scale = act_scale[g] * weight_scale[(row_offset + n) * scale_stride_m + g * scale_stride_g] / weight_global;
    acc += e2m1_from_code(a_code) * e2m1_from_code(b_code) * scale;
  }
  return acc;
}

__global__ void p89_per16_scale_kernel(
    const uint8_t* __restrict__ act,
    const float* __restrict__ act_scale,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scale,
    const float* __restrict__ weight_global_scale,
    float* __restrict__ out,
    int row_offset,
    int k_offset,
    int packed_stride_m,
    int scale_stride_m,
    int scale_stride_g,
    uint8_t scale_byte) {
  const int lane = threadIdx.x & 31;
  const int block = blockIdx.x;

  float p0[4];
  float p1[4];
  run_mma_group(act, weight, row_offset, k_offset, packed_stride_m, 0, lane, scale_byte, p0);
  run_mma_group(act, weight, row_offset, k_offset, packed_stride_m, 1, lane, scale_byte, p1);

  const float weight_global = weight_global_scale[0];
  const int g0 = k_offset >> 4;
  const int g1 = g0 + 1;
  const int64_t base = (static_cast<int64_t>(block) * 32 + lane) * 4 * 10;
  #pragma unroll
  for (int v = 0; v < 4; ++v) {
    int m = 0;
    int n = 0;
    c_coord_from_lane_value(lane, v, &m, &n);
    const float s0 = act_scale[g0] * weight_scale[(row_offset + n) * scale_stride_m + g0 * scale_stride_g] / weight_global;
    const float s1 = act_scale[g1] * weight_scale[(row_offset + n) * scale_stride_m + g1 * scale_stride_g] / weight_global;
    const float exact_split = p0[v] * s0 + p1[v] * s1;
    const float expected = scalar_scaled_reference(
        act, act_scale, weight, weight_scale, weight_global, row_offset, k_offset,
        packed_stride_m, scale_stride_m, scale_stride_g, n);
    out[base + v * 10 + 0] = exact_split;
    out[base + v * 10 + 1] = expected;
    out[base + v * 10 + 2] = p0[v];
    out[base + v * 10 + 3] = p1[v];
    out[base + v * 10 + 4] = s0;
    out[base + v * 10 + 5] = s1;
    out[base + v * 10 + 6] = static_cast<float>(m);
    out[base + v * 10 + 7] = static_cast<float>(n);
    out[base + v * 10 + 8] = p0[v] + p1[v];
    out[base + v * 10 + 9] = (s0 + s1) * 0.5f;
  }
}

}  // namespace

torch::Tensor p89_per16_scale_tile_contract(
    torch::Tensor act_packed,
    torch::Tensor act_scale,
    torch::Tensor weight_packed,
    torch::Tensor weight_scale,
    torch::Tensor weight_global_scale,
    int64_t row_offset,
    int64_t k_offset,
    int64_t scale_byte) {
  TORCH_CHECK(act_packed.is_cuda(), "act_packed must be CUDA");
  TORCH_CHECK(act_scale.is_cuda(), "act_scale must be CUDA");
  TORCH_CHECK(weight_packed.is_cuda(), "weight_packed must be CUDA");
  TORCH_CHECK(weight_scale.is_cuda(), "weight_scale must be CUDA");
  TORCH_CHECK(weight_global_scale.is_cuda(), "weight_global_scale must be CUDA");
  TORCH_CHECK(act_packed.scalar_type() == torch::kUInt8, "act_packed must be uint8");
  TORCH_CHECK(act_scale.scalar_type() == torch::kFloat32, "act_scale must be float32");
  TORCH_CHECK(weight_packed.scalar_type() == torch::kUInt8, "weight_packed must be uint8");
  TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32, "weight_scale must be float32");
  TORCH_CHECK(weight_global_scale.scalar_type() == torch::kFloat32, "weight_global_scale must be float32");
  TORCH_CHECK(act_packed.dim() == 1, "act_packed must be [K/2]");
  TORCH_CHECK(act_scale.dim() == 1, "act_scale must be [K/16]");
  TORCH_CHECK(weight_packed.dim() == 2, "weight_packed must be [rows,K/2]");
  TORCH_CHECK(weight_scale.dim() == 2, "weight_scale must be [rows,K/16]");
  TORCH_CHECK(row_offset >= 0 && row_offset + 8 <= weight_packed.size(0), "row_offset out of bounds");
  TORCH_CHECK(k_offset >= 0 && k_offset + 32 <= act_packed.size(0) * 2, "k_offset out of bounds");
  TORCH_CHECK((k_offset % 32) == 0, "k_offset must be divisible by 32 for P89 two-group tile");
  TORCH_CHECK(scale_byte >= 0 && scale_byte <= 255, "scale_byte must be a byte");
  auto act_c = act_packed.contiguous();
  auto act_scale_c = act_scale.contiguous();
  auto weight_c = weight_packed.contiguous();
  auto weight_scale_c = weight_scale.contiguous();
  auto weight_global_c = weight_global_scale.contiguous();
  auto out = torch::empty({1, 32, 4, 10}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  p89_per16_scale_kernel<<<1, 32>>>(
      act_c.data_ptr<uint8_t>(),
      act_scale_c.data_ptr<float>(),
      weight_c.data_ptr<uint8_t>(),
      weight_scale_c.data_ptr<float>(),
      weight_global_c.data_ptr<float>(),
      out.data_ptr<float>(),
      static_cast<int>(row_offset),
      static_cast<int>(k_offset),
      static_cast<int>(weight_c.stride(0)),
      static_cast<int>(weight_scale_c.stride(0)),
      static_cast<int>(weight_scale_c.stride(1)),
      static_cast<uint8_t>(scale_byte));
  const cudaError_t launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess, "p89_per16_scale_kernel launch failed: ", cudaGetErrorString(launch_err));
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
    cpp_path = build_root / "p89_bindings.cpp"
    cu_path = build_root / "p89_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")
    return load(
        name="lynn_native_p89_sm120a_per16_scale_tile_contract",
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


def _fold_errors(cpu: torch.Tensor) -> dict[str, dict[str, float]]:
    expected = cpu[..., 1]
    p0 = cpu[..., 2]
    p1 = cpu[..., 3]
    s0 = cpu[..., 4]
    s1 = cpu[..., 5]
    code_sum = cpu[..., 8]
    modes = {
        "mean": (s0 + s1) * 0.5,
        "max": torch.maximum(s0, s1),
        "min": torch.minimum(s0, s1),
        "geom": torch.sqrt(torch.clamp(s0 * s1, min=0.0)),
        "weighted_abs": (p0.abs() * s0 + p1.abs() * s1) / torch.clamp(p0.abs() + p1.abs(), min=1e-12),
    }
    rows: dict[str, dict[str, float]] = {}
    for name, scale in modes.items():
        approx = code_sum * scale
        err = (approx - expected).abs()
        rows[name] = {
            "max_abs": float(err.max().item()),
            "mean_abs": float(err.float().mean().item()),
            "rel_l2": float(torch.linalg.vector_norm(approx - expected).item() / max(torch.linalg.vector_norm(expected).item(), 1e-12)),
        }
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p89_sm120a_per16_scale_tile_contract")
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
        raise RuntimeError("P89 requires CUDA")
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
    expert_id = int(expert_ids[args.expert_slot].item())

    gate_up_packed, gate_up_scale, gate_up_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{args.layer}.mlp.experts.gate_up_proj",
        runner.device,
    )
    act_packed, act_scale = _quantize_activation_to_fp4(hidden_2d)
    act_packed_1d = act_packed[0].contiguous()
    act_scale_1d = act_scale[0].float().contiguous()
    weight_packed = gate_up_packed[expert_id].contiguous()
    weight_scale = gate_up_scale[expert_id].float().contiguous()

    t0 = time.time()
    module = _build_module(Path(args.build_dir), args.verbose)
    build_s = time.time() - t0

    def run_tile() -> torch.Tensor:
        return module.per16_scale_tile_contract(
            act_packed_1d,
            act_scale_1d,
            weight_packed,
            weight_scale,
            gate_up_global.float(),
            args.row_offset,
            args.k_offset,
            args.scale_byte,
        )

    out, times_ms = _time(run_tile, args.repeats, args.warmup)
    cpu = out.detach().cpu()
    exact = cpu[..., 0]
    expected = cpu[..., 1]
    err = (exact - expected).abs()
    max_idx = int(err.reshape(-1).argmax().item())
    flat = cpu.reshape(-1, 10)
    worst = flat[max_idx]

    result = {
        "schema_version": "lynn-engine-p89-sm120a-per16-scale-tile-contract-v1",
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
        "split16_exact": {
            "max_abs_err": float(err.max().item()),
            "mean_abs_err": float(err.float().mean().item()),
            "all_exact": bool((err == 0).all().item()),
            "rel_l2": float(torch.linalg.vector_norm(exact - expected).item() / max(torch.linalg.vector_norm(expected).item(), 1e-12)),
            "observed_min": float(exact.min().item()),
            "observed_max": float(exact.max().item()),
            "expected_min": float(expected.min().item()),
            "expected_max": float(expected.max().item()),
            "worst": {
                "observed": float(worst[0].item()),
                "expected": float(worst[1].item()),
                "partial0": float(worst[2].item()),
                "partial1": float(worst[3].item()),
                "scale0": float(worst[4].item()),
                "scale1": float(worst[5].item()),
                "m": int(worst[6].item()),
                "n": int(worst[7].item()),
            },
        },
        "single_k32_fold_errors": _fold_errors(cpu),
        "sample_block0_lane0": [
            {
                "exact": float(cpu[0, 0, v, 0].item()),
                "expected": float(cpu[0, 0, v, 1].item()),
                "partial0": float(cpu[0, 0, v, 2].item()),
                "partial1": float(cpu[0, 0, v, 3].item()),
                "scale0": float(cpu[0, 0, v, 4].item()),
                "scale1": float(cpu[0, 0, v, 5].item()),
                "m": int(cpu[0, 0, v, 6].item()),
                "n": int(cpu[0, 0, v, 7].item()),
            }
            for v in range(4)
        ],
        "decision": (
            "Current Lynn per-16 scale contract is exactly consumable by split16 neutral-scale MMA plus explicit per-group scale accumulation."
            if bool((err == 0).all().item())
            else "Split16 strategy mismatches scalar scaled reference; inspect scale/indexing before kernel construction."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
