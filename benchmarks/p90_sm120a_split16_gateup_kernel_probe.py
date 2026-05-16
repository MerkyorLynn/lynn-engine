#!/usr/bin/env python3
"""P90: first real split16 per-16 SM120a FP4 gate/up kernel probe.

P89 proved the tile math for a single K=32 gate/up tile. P90 stretches that
contract into the first useful gate/up shape:

* one real routed expert;
* eight consecutive gate rows and eight consecutive up rows;
* full hidden K=2048;
* current Lynn-native per-16 activation/weight scales;
* SM120a blockscaled FP4 MMA with two K16 passes per K32 tile.

This is still a contract probe, not the final production active-MoE kernel. The
goal is to prove the current Lynn-native artifact can drive a real gate/up row
tile without re-quantization.
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

torch::Tensor p90_split16_gateup_kernel(
    torch::Tensor act_packed,
    torch::Tensor act_scale,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    int64_t inter_offset,
    int64_t scale_byte);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("split16_gateup_kernel", &p90_split16_gateup_kernel,
        "P90 split16 per-16 SM120a FP4 gate/up row-tile kernel");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace {

constexpr int kHidden = 2048;
constexpr int kIntermediate = 512;
constexpr int kRows = 8;

__device__ __forceinline__ uint8_t get_code_from_packed(const uint8_t* ptr, int elem) {
  const uint8_t byte = ptr[elem >> 1];
  return (elem & 1) == 0 ? (byte & 0x0Fu) : ((byte >> 4) & 0x0Fu);
}

__device__ __forceinline__ uint32_t mma_byte_from_code(uint8_t code) {
  // CuTe float_e2m1_t stores the 4-bit code in bits [5:2].
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
    const uint8_t* weight,
    int row_offset,
    int k_offset,
    int packed_stride_m,
    int group16,
    int lane,
    uint32_t* words) {
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

__device__ __forceinline__ void accumulate_rows(
    const float* p0,
    const float* p1,
    const float* act_scale,
    const float* weight_scale,
    float weight_global,
    int row_offset,
    int k_offset,
    int scale_stride_m,
    int scale_stride_g,
    int lane,
    float* out) {
  const int g0 = k_offset >> 4;
  const int g1 = g0 + 1;
  #pragma unroll
  for (int v = 0; v < 4; ++v) {
    int m = 0;
    int n = 0;
    c_coord_from_lane_value(lane, v, &m, &n);
    if (m == 0) {
      const float s0 = act_scale[g0] * weight_scale[(row_offset + n) * scale_stride_m + g0 * scale_stride_g] / weight_global;
      const float s1 = act_scale[g1] * weight_scale[(row_offset + n) * scale_stride_m + g1 * scale_stride_g] / weight_global;
      atomicAdd(out + n, p0[v] * s0 + p1[v] * s1);
    }
  }
}

__global__ void p90_split16_gateup_kernel_impl(
    const uint8_t* __restrict__ act,
    const float* __restrict__ act_scale,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scale,
    const float* __restrict__ weight_global_scale,
    float* __restrict__ out,
    int inter_offset,
    int packed_stride_m,
    int scale_stride_m,
    int scale_stride_g,
    uint8_t scale_byte) {
  const int lane = threadIdx.x & 31;
  const float weight_global = weight_global_scale[0];
  float* gate_out = out;
  float* up_out = out + kRows;

  for (int k_offset = 0; k_offset < kHidden; k_offset += 32) {
    float g0[4];
    float g1[4];
    float u0[4];
    float u1[4];
    run_mma_group(act, weight, inter_offset, k_offset, packed_stride_m, 0, lane, scale_byte, g0);
    run_mma_group(act, weight, inter_offset, k_offset, packed_stride_m, 1, lane, scale_byte, g1);
    run_mma_group(act, weight, kIntermediate + inter_offset, k_offset, packed_stride_m, 0, lane, scale_byte, u0);
    run_mma_group(act, weight, kIntermediate + inter_offset, k_offset, packed_stride_m, 1, lane, scale_byte, u1);
    accumulate_rows(g0, g1, act_scale, weight_scale, weight_global, inter_offset, k_offset,
                    scale_stride_m, scale_stride_g, lane, gate_out);
    accumulate_rows(u0, u1, act_scale, weight_scale, weight_global, kIntermediate + inter_offset, k_offset,
                    scale_stride_m, scale_stride_g, lane, up_out);
  }
}

}  // namespace

torch::Tensor p90_split16_gateup_kernel(
    torch::Tensor act_packed,
    torch::Tensor act_scale,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    int64_t inter_offset,
    int64_t scale_byte) {
  TORCH_CHECK(act_packed.is_cuda(), "act_packed must be CUDA");
  TORCH_CHECK(act_scale.is_cuda(), "act_scale must be CUDA");
  TORCH_CHECK(gate_up_packed.is_cuda(), "gate_up_packed must be CUDA");
  TORCH_CHECK(gate_up_scale.is_cuda(), "gate_up_scale must be CUDA");
  TORCH_CHECK(gate_up_global_scale.is_cuda(), "gate_up_global_scale must be CUDA");
  TORCH_CHECK(act_packed.scalar_type() == torch::kUInt8, "act_packed must be uint8");
  TORCH_CHECK(act_scale.scalar_type() == torch::kFloat32, "act_scale must be float32");
  TORCH_CHECK(gate_up_packed.scalar_type() == torch::kUInt8, "gate_up_packed must be uint8");
  TORCH_CHECK(gate_up_scale.scalar_type() == torch::kFloat32, "gate_up_scale must be float32");
  TORCH_CHECK(gate_up_global_scale.scalar_type() == torch::kFloat32, "gate_up_global_scale must be float32");
  TORCH_CHECK(act_packed.dim() == 1 && act_packed.numel() == kHidden / 2, "act_packed must be [1024]");
  TORCH_CHECK(act_scale.dim() == 1 && act_scale.numel() == kHidden / 16, "act_scale must be [128]");
  TORCH_CHECK(gate_up_packed.dim() == 2, "gate_up_packed must be [1024,1024] for one expert");
  TORCH_CHECK(gate_up_scale.dim() == 2, "gate_up_scale must be [1024,128] for one expert");
  TORCH_CHECK(inter_offset >= 0 && inter_offset + kRows <= kIntermediate, "inter_offset out of bounds");
  TORCH_CHECK(scale_byte >= 0 && scale_byte <= 255, "scale_byte must be a byte");

  auto act_c = act_packed.contiguous();
  auto act_scale_c = act_scale.contiguous();
  auto weight_c = gate_up_packed.contiguous();
  auto weight_scale_c = gate_up_scale.contiguous();
  auto global_c = gate_up_global_scale.contiguous();
  auto out = torch::zeros({2, kRows}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

  p90_split16_gateup_kernel_impl<<<1, 32>>>(
      act_c.data_ptr<uint8_t>(),
      act_scale_c.data_ptr<float>(),
      weight_c.data_ptr<uint8_t>(),
      weight_scale_c.data_ptr<float>(),
      global_c.data_ptr<float>(),
      out.data_ptr<float>(),
      static_cast<int>(inter_offset),
      static_cast<int>(weight_c.stride(0)),
      static_cast<int>(weight_scale_c.stride(0)),
      static_cast<int>(weight_scale_c.stride(1)),
      static_cast<uint8_t>(scale_byte));
  const cudaError_t launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess, "p90_split16_gateup_kernel launch failed: ", cudaGetErrorString(launch_err));
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
    cpp_path = build_root / "p90_bindings.cpp"
    cu_path = build_root / "p90_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")
    return load(
        name="lynn_native_p90_sm120a_split16_gateup_kernel",
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


def _unpack_codes(packed: torch.Tensor, elems: int) -> torch.Tensor:
    packed_cpu = packed.detach().cpu().to(torch.uint8).reshape(-1)
    low = packed_cpu & 0x0F
    high = (packed_cpu >> 4) & 0x0F
    codes = torch.empty((packed_cpu.numel() * 2,), dtype=torch.long)
    codes[0::2] = low.to(torch.long)
    codes[1::2] = high.to(torch.long)
    return codes[:elems]


def _e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )
    return table[codes.to(torch.long)]


def _reference_rows(
    act_packed: torch.Tensor,
    act_scale: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global: torch.Tensor,
    inter_offset: int,
) -> torch.Tensor:
    act_codes = _unpack_codes(act_packed, 2048)
    act_vals = _e2m1_values(act_codes) * act_scale.detach().cpu().float().repeat_interleave(16)
    packed_cpu = weight_packed.detach().cpu().contiguous()
    scale_cpu = weight_scale.detach().cpu().float().contiguous()
    global_val = float(weight_global.detach().cpu().float().reshape(-1)[0].item())
    out = torch.empty((2, 8), dtype=torch.float32)
    for half, row_base in enumerate((inter_offset, 512 + inter_offset)):
        for n in range(8):
            row = row_base + n
            row_codes = _unpack_codes(packed_cpu[row], 2048)
            row_vals = _e2m1_values(row_codes) * (scale_cpu[row].repeat_interleave(16) / global_val)
            out[half, n] = torch.sum(act_vals * row_vals)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p90_sm120a_split16_gateup_kernel")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--expert-slot", type=int, default=0)
    ap.add_argument("--inter-offset", type=int, default=0)
    ap.add_argument("--scale-byte", type=int, default=127)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P90 requires CUDA")
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
        return module.split16_gateup_kernel(
            act_packed_1d,
            act_scale_1d,
            weight_packed,
            weight_scale,
            gate_up_global.float(),
            args.inter_offset,
            args.scale_byte,
        )

    observed, times_ms = _time(run_tile, args.repeats, args.warmup)
    expected = _reference_rows(
        act_packed_1d,
        act_scale_1d,
        weight_packed,
        weight_scale,
        gate_up_global,
        args.inter_offset,
    ).to(observed.device)
    err = (observed - expected).abs()
    max_idx = int(err.reshape(-1).argmax().item())
    worst_half = max_idx // 8
    worst_row = max_idx % 8

    result = {
        "schema_version": "lynn-engine-p90-sm120a-split16-gateup-kernel-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "expert_slot": args.expert_slot,
        "expert_id": expert_id,
        "top_k_expert_ids": [int(x) for x in expert_ids.tolist()],
        "inter_offset": args.inter_offset,
        "rows_per_tile": 8,
        "scale_byte": args.scale_byte,
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
        "contract": {
            "max_abs_err": float(err.max().item()),
            "mean_abs_err": float(err.float().mean().item()),
            "rel_l2": float(torch.linalg.vector_norm(observed - expected).item() / max(torch.linalg.vector_norm(expected).item(), 1e-12)),
            "pass_with_tolerance": bool(err.max().item() <= 1e-5),
            "tolerance": 1e-5,
            "observed_min": float(observed.min().item()),
            "observed_max": float(observed.max().item()),
            "expected_min": float(expected.min().item()),
            "expected_max": float(expected.max().item()),
            "worst": {
                "half": "gate" if worst_half == 0 else "up",
                "row": int(args.inter_offset + worst_row),
                "observed": float(observed.reshape(-1)[max_idx].item()),
                "expected": float(expected.reshape(-1)[max_idx].item()),
            },
        },
        "observed": observed.detach().cpu().tolist(),
        "expected": expected.detach().cpu().tolist(),
        "decision": (
            "PASS: first real split16 per-16 SM120a FP4 gate/up row tile consumes the current Lynn-native artifact within tolerance."
            if bool(err.max().item() <= 1e-5)
            else "FAIL: split16 gate/up row tile diverged; inspect MMA fragment ownership or scale indexing."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
