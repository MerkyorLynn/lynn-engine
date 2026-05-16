#!/usr/bin/env python3
"""P93: top-k split16 per-16 SM120a FP4 gate/up backend probe.

P92 proved a full 512-row gate/up projection for one routed expert. P93 turns
that into the first production-shaped gate/up backend candidate:

* one CUDA launch;
* grid.x = top_k experts, grid.y = 64 row tiles;
* each block owns 8 intermediate rows for one expert slot;
* shared-memory accumulation avoids P92 global atomics;
* output is `inter[top_k,512]` BF16, ready for the down projection.

This still does not touch runtime defaults. It is a contract and shape probe for
the current Lynn-native NVFP4 artifact.
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
from triton_kernels.nvfp4_moe import nvfp4_grouped_gate_up_silu_fast_decode  # noqa: E402


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor p93_split16_gateup_topk(
    torch::Tensor act_packed,
    torch::Tensor act_scale,
    torch::Tensor expert_ids,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    int64_t scale_byte);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("split16_gateup_topk", &p93_split16_gateup_topk,
        "P93 top-k split16 per-16 SM120a FP4 gate/up backend probe");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <cuda_bf16.h>
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

__device__ __forceinline__ void accumulate_rows_shared(
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
    float* shared_rows) {
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
      atomicAdd(shared_rows + n, p0[v] * s0 + p1[v] * s1);
    }
  }
}

__global__ void p93_split16_gateup_topk_kernel(
    const uint8_t* __restrict__ act,
    const float* __restrict__ act_scale,
    const int32_t* __restrict__ expert_ids,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scale,
    const float* __restrict__ weight_global_scale,
    __nv_bfloat16* __restrict__ inter,
    int top_k,
    int packed_stride_e,
    int packed_stride_m,
    int packed_stride_n,
    int scale_stride_e,
    int scale_stride_m,
    int scale_stride_g,
    int inter_stride_k,
    int inter_stride_i,
    uint8_t scale_byte) {
  const int lane = threadIdx.x & 31;
  const int slot = blockIdx.x;
  const int tile = blockIdx.y;
  if (slot >= top_k) {
    return;
  }
  const int expert = expert_ids[slot];
  const int row_base = tile * kRows;
  const uint8_t* expert_weight = weight + static_cast<int64_t>(expert) * packed_stride_e;
  const float* expert_scale = weight_scale + static_cast<int64_t>(expert) * scale_stride_e;
  const float weight_global = weight_global_scale[0];

  __shared__ float gate_s[kRows];
  __shared__ float up_s[kRows];
  if (lane < kRows) {
    gate_s[lane] = 0.0f;
    up_s[lane] = 0.0f;
  }
  __syncthreads();

  for (int k_offset = 0; k_offset < kHidden; k_offset += 32) {
    float g0[4];
    float g1[4];
    float u0[4];
    float u1[4];
    run_mma_group(act, expert_weight, row_base, k_offset, packed_stride_m, 0, lane, scale_byte, g0);
    run_mma_group(act, expert_weight, row_base, k_offset, packed_stride_m, 1, lane, scale_byte, g1);
    run_mma_group(act, expert_weight, kIntermediate + row_base, k_offset, packed_stride_m, 0, lane, scale_byte, u0);
    run_mma_group(act, expert_weight, kIntermediate + row_base, k_offset, packed_stride_m, 1, lane, scale_byte, u1);
    accumulate_rows_shared(g0, g1, act_scale, expert_scale, weight_global, row_base, k_offset, scale_stride_m, scale_stride_g, lane, gate_s);
    accumulate_rows_shared(u0, u1, act_scale, expert_scale, weight_global, kIntermediate + row_base, k_offset, scale_stride_m, scale_stride_g, lane, up_s);
  }

  __syncthreads();
  if (lane < kRows) {
    const int row = row_base + lane;
    const float gate = gate_s[lane];
    const float up = up_s[lane];
    const float silu = gate / (1.0f + expf(-gate));
    inter[static_cast<int64_t>(slot) * inter_stride_k + static_cast<int64_t>(row) * inter_stride_i] =
        __float2bfloat16(silu * up);
  }
}

}  // namespace

torch::Tensor p93_split16_gateup_topk(
    torch::Tensor act_packed,
    torch::Tensor act_scale,
    torch::Tensor expert_ids,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    int64_t scale_byte) {
  TORCH_CHECK(act_packed.is_cuda(), "act_packed must be CUDA");
  TORCH_CHECK(act_scale.is_cuda(), "act_scale must be CUDA");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be CUDA");
  TORCH_CHECK(gate_up_packed.is_cuda(), "gate_up_packed must be CUDA");
  TORCH_CHECK(gate_up_scale.is_cuda(), "gate_up_scale must be CUDA");
  TORCH_CHECK(gate_up_global_scale.is_cuda(), "gate_up_global_scale must be CUDA");
  TORCH_CHECK(act_packed.scalar_type() == torch::kUInt8, "act_packed must be uint8");
  TORCH_CHECK(act_scale.scalar_type() == torch::kFloat32, "act_scale must be float32");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(gate_up_packed.scalar_type() == torch::kUInt8, "gate_up_packed must be uint8");
  TORCH_CHECK(gate_up_scale.scalar_type() == torch::kFloat32, "gate_up_scale must be float32");
  TORCH_CHECK(gate_up_global_scale.scalar_type() == torch::kFloat32, "gate_up_global_scale must be float32");
  TORCH_CHECK(act_packed.dim() == 1 && act_packed.numel() == kHidden / 2, "act_packed must be [1024]");
  TORCH_CHECK(act_scale.dim() == 1 && act_scale.numel() == kHidden / 16, "act_scale must be [128]");
  TORCH_CHECK(expert_ids.dim() == 1, "expert_ids must be [top_k]");
  TORCH_CHECK(gate_up_packed.dim() == 3, "gate_up_packed must be [experts,1024,1024]");
  TORCH_CHECK(gate_up_scale.dim() == 3, "gate_up_scale must be [experts,1024,128]");
  TORCH_CHECK(scale_byte >= 0 && scale_byte <= 255, "scale_byte must be a byte");

  auto act_c = act_packed.contiguous();
  auto act_scale_c = act_scale.contiguous();
  auto expert_ids_c = expert_ids.contiguous();
  auto weight_c = gate_up_packed.contiguous();
  auto weight_scale_c = gate_up_scale.contiguous();
  auto global_c = gate_up_global_scale.contiguous();
  const int64_t top_k = expert_ids_c.numel();
  auto out = torch::empty({top_k, kIntermediate}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));

  const dim3 grid(static_cast<unsigned int>(top_k), kIntermediate / kRows);
  p93_split16_gateup_topk_kernel<<<grid, 32>>>(
      act_c.data_ptr<uint8_t>(),
      act_scale_c.data_ptr<float>(),
      expert_ids_c.data_ptr<int32_t>(),
      weight_c.data_ptr<uint8_t>(),
      weight_scale_c.data_ptr<float>(),
      global_c.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(top_k),
      static_cast<int>(weight_c.stride(0)),
      static_cast<int>(weight_c.stride(1)),
      static_cast<int>(weight_c.stride(2)),
      static_cast<int>(weight_scale_c.stride(0)),
      static_cast<int>(weight_scale_c.stride(1)),
      static_cast<int>(weight_scale_c.stride(2)),
      static_cast<int>(out.stride(0)),
      static_cast<int>(out.stride(1)),
      static_cast<uint8_t>(scale_byte));
  const cudaError_t launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess, "p93_split16_gateup_topk_kernel launch failed: ", cudaGetErrorString(launch_err));
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
    cpp_path = build_root / "p93_bindings.cpp"
    cu_path = build_root / "p93_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")
    return load(
        name="lynn_native_p93_sm120a_split16_gateup_topk_backend",
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


def _quantized_activation_reference(
    act_packed: torch.Tensor,
    act_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global: torch.Tensor,
) -> torch.Tensor:
    act_codes = _unpack_codes(act_packed, 2048)
    act_vals = _e2m1_values(act_codes) * act_scale.detach().cpu().float().repeat_interleave(16)
    packed_cpu = weight_packed.detach().cpu().contiguous()
    scale_cpu = weight_scale.detach().cpu().float().contiguous()
    expert_ids_cpu = expert_ids.detach().cpu().to(torch.long)
    global_val = float(weight_global.detach().cpu().float().reshape(-1)[0].item())
    out = torch.empty((expert_ids_cpu.numel(), 512), dtype=torch.float32)
    for slot, expert in enumerate(expert_ids_cpu.tolist()):
        for inter in range(512):
            gate_codes = _unpack_codes(packed_cpu[expert, inter], 2048)
            up_codes = _unpack_codes(packed_cpu[expert, 512 + inter], 2048)
            gate_vals = _e2m1_values(gate_codes) * (scale_cpu[expert, inter].repeat_interleave(16) / global_val)
            up_vals = _e2m1_values(up_codes) * (scale_cpu[expert, 512 + inter].repeat_interleave(16) / global_val)
            gate = torch.sum(act_vals * gate_vals)
            up = torch.sum(act_vals * up_vals)
            out[slot, inter] = (gate * torch.sigmoid(gate)) * up
    return out


def _diff(ref: torch.Tensor, out: torch.Tensor) -> dict[str, float]:
    rf = ref.float().reshape(-1)
    of = out.float().reshape(-1)
    delta = of - rf
    denom = torch.linalg.vector_norm(rf).clamp_min(1e-20)
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rel_l2": float((torch.linalg.vector_norm(delta) / denom).item()),
        "cosine": float(F.cosine_similarity(rf, of, dim=0).item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p93_sm120a_split16_gateup_topk_backend")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--scale-byte", type=int, default=127)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P93 requires CUDA")
    os.environ.setdefault("LYNN_NATIVE_CUDA_ARCH", "sm_120a")
    _prepare_path()

    model_dir = Path(args.model)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    h_layer, _ = _prefill_to_layer_input(runner, args.layer, args.prompt)
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    h_flat = h_moe.reshape(-1, h_moe.shape[-1])
    hidden = h_flat[0].contiguous()
    top_k = int(cfg["num_experts_per_tok"])
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    _, expert_indices = torch.topk(router_logits, top_k, dim=-1, sorted=False)
    expert_ids = expert_indices[0].to(torch.int32).contiguous()

    gate_up_packed, gate_up_scale, gate_up_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{args.layer}.mlp.experts.gate_up_proj",
        runner.device,
    )
    act_packed, act_scale = _quantize_activation_to_fp4(hidden.view(1, -1))
    act_packed_1d = act_packed[0].contiguous()
    act_scale_1d = act_scale[0].float().contiguous()

    t0 = time.time()
    module = _build_module(Path(args.build_dir), args.verbose)
    build_s = time.time() - t0

    def native_gateup() -> torch.Tensor:
        return module.split16_gateup_topk(
            act_packed_1d,
            act_scale_1d,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global.float(),
            args.scale_byte,
        )

    def triton_gateup() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu_fast_decode(
            hidden,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )

    native, native_times = _time(native_gateup, args.repeats, args.warmup)
    triton, triton_times = _time(triton_gateup, args.repeats, args.warmup)
    ref_t0 = time.time()
    quantized_ref = _quantized_activation_reference(
        act_packed_1d,
        act_scale_1d,
        expert_ids,
        gate_up_packed,
        gate_up_scale,
        gate_up_global,
    ).to(native.device)
    ref_seconds = time.time() - ref_t0

    diff_quantized_ref = _diff(quantized_ref, native)
    diff_triton_bf16_act = _diff(triton, native)
    result = {
        "schema_version": "lynn-engine-p93-sm120a-split16-gateup-topk-backend-v1",
        "model": args.model,
        "layer": args.layer,
        "prompt": args.prompt,
        "top_k": top_k,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "scale_byte": args.scale_byte,
        "torch": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "include_paths": discover_native_include_paths(),
        "cuda_cflags": native_cuda_extra_cuda_cflags(),
        "build_seconds": build_s,
        "reference_seconds": ref_seconds,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "native_times_ms": native_times,
        "native_mean_ms": statistics.fmean(native_times),
        "native_median_ms": statistics.median(native_times),
        "triton_times_ms": triton_times,
        "triton_mean_ms": statistics.fmean(triton_times),
        "triton_median_ms": statistics.median(triton_times),
        "native_vs_triton_speedup_median": float(statistics.median(triton_times) / statistics.median(native_times)),
        "diff_native_vs_quantized_activation_reference": diff_quantized_ref,
        "diff_native_vs_current_triton_bf16_activation": diff_triton_bf16_act,
        "contract_pass": bool(
            diff_quantized_ref["rel_l2"] <= 0.01
            and diff_quantized_ref["cosine"] >= 0.9999
            and diff_quantized_ref["max_abs"] <= 0.05
        ),
        "runtime_promote": False,
        "decision": (
            "PASS: top-k split16 gate/up backend matches the quantized-activation reference; next gate is active MoE composition."
            if diff_quantized_ref["rel_l2"] <= 0.01
            else "FAIL: top-k split16 gate/up backend diverged from quantized-activation reference."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["contract_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
