"""Spark sm_121 FP8 native active-MoE module.

Self-contained module that JIT-builds the SP-12-E-v2 FP8 gate_up + down
kernels and provides a Python-level `active_moe_spark_fp8` function that
matches the existing Triton path's signature.

Built once per process. Subsequent calls use the cached extension.

Activated when `LYNN_NATIVE_ACTIVE_MOE_BACKEND=spark_fp8`. The dispatch is
added in `engine/moe_packed_nvfp4.py`.

Numerical contract: identical to SP-12-D/E probes (max_abs_err ~1e-6 vs
quantized-activation scalar reference; same numerical class as Codex P90
on R6000 sm_120a FP4 path — both consume identical Lynn-native artifact).

Performance contract: SP-12-E-v2 standalone microbench measures 98.73 us
total (gate_up + down) for active-MoE forward at hidden=2048,
intermediate=512, num_experts=256, top_k=8. Per-decode-step contribution
for Lynn 27B: 40 × 98.73 us = 3.95 ms. Net TPS gain over current Triton
SP-08 path depends on what fraction of decode step Triton MoE currently
takes; estimated 5-10ms saved per step = +15-50% TPS.
"""
from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from pathlib import Path

import torch
import torch.nn.functional as F


_EXTENSION = None
_BUILD_DIR_DEFAULT = "/tmp/lynn_spark_fp8_build"

# E2M1 representable magnitudes (for batched quantization helper)
_E2M1_TABLE_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_e2m1_table(device: torch.device) -> torch.Tensor:
    if device not in _E2M1_TABLE_CACHE:
        _E2M1_TABLE_CACHE[device] = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.float32, device=device,
        )
    return _E2M1_TABLE_CACHE[device]


def _quantize_e2m1_batched(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [N, K] (or [K]) tensor to packed E2M1 + per-16 FP32 scale.

    Lynn's `quantize_fp4_m1_native` only accepts [K] or [1,K]. The inter
    tensor after SiLU(gate)*up has shape [top_k, intermediate] which needs
    a batched quantizer.

    Returns: (packed [N, K/2] uint8, scale [N, K/16] float32)
    """
    flat = x.dim() == 1
    if flat:
        x = x.unsqueeze(0)
    N, K = x.shape
    assert K % 16 == 0, f"K={K} must be multiple of 16"

    table = _get_e2m1_table(x.device)
    xg = x.float().reshape(N, K // 16, 16)
    abs_max = xg.abs().amax(dim=-1)
    scale = (abs_max / 6.0).clamp_min(1e-8)
    normalized = (xg.abs() / scale.unsqueeze(-1)).clamp(0, 6.0)
    mag = torch.argmin((normalized.unsqueeze(-1) - table.view(1, 1, 1, -1)).abs(), dim=-1)
    sign = (xg < 0).to(torch.uint8) * 8
    codes = (mag.to(torch.uint8) | sign).reshape(N, K)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    scale = scale.contiguous()
    if flat:
        packed = packed.squeeze(0)
        scale = scale.squeeze(0)
    return packed, scale


_CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor sparkfp8_gate_up(
    torch::Tensor act_packed,
    torch::Tensor act_scale,
    torch::Tensor expert_ids,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    int64_t intermediate
);

torch::Tensor sparkfp8_down(
    torch::Tensor inter_packed,
    torch::Tensor inter_scale,
    torch::Tensor expert_ids,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale,
    int64_t hidden
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gate_up", &sparkfp8_gate_up, "Spark sm_121 FP8 gate_up active MoE");
    m.def("down", &sparkfp8_down, "Spark sm_121 FP8 down active MoE");
}
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr uint32_t LUT_REG_0 = 0x3C383000;
constexpr uint32_t LUT_REG_1 = 0x4C484440;
constexpr uint32_t LUT_REG_2 = 0xBCB8B080;
constexpr uint32_t LUT_REG_3 = 0xCCC8C4C0;

__device__ __forceinline__ uint8_t lut_nibble_to_fp8(uint8_t nibble) {
    uint8_t bi = nibble & 0x0F;
    uint32_t reg;
    if (bi < 4)        reg = LUT_REG_0;
    else if (bi < 8)   reg = LUT_REG_1;
    else if (bi < 12)  reg = LUT_REG_2;
    else               reg = LUT_REG_3;
    uint8_t shift = (bi & 3) * 8;
    return (uint8_t)((reg >> shift) & 0xFF);
}

__device__ __forceinline__ uint32_t load_4_nibbles_as_fp8(const uint8_t* ptr, int byte_offset) {
    uint16_t bytes = *reinterpret_cast<const uint16_t*>(ptr + byte_offset);
    uint8_t n0 = (uint8_t)(bytes & 0x0F);
    uint8_t n1 = (uint8_t)((bytes >> 4) & 0x0F);
    uint8_t n2 = (uint8_t)((bytes >> 8) & 0x0F);
    uint8_t n3 = (uint8_t)((bytes >> 12) & 0x0F);
    uint8_t b0 = lut_nibble_to_fp8(n0);
    uint8_t b1 = lut_nibble_to_fp8(n1);
    uint8_t b2 = lut_nibble_to_fp8(n2);
    uint8_t b3 = lut_nibble_to_fp8(n3);
    return (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
}

__device__ __forceinline__ void fp8_mma_m16n8k32(
    const uint32_t a[4], const uint32_t b[2],
    float d[4]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f)
    );
}

__device__ __forceinline__ void fill_a_two_halves_vec(
    const uint8_t* act_packed,
    int k_base, int lane,
    uint32_t a_low[4], uint32_t a_high[4]
) {
    const int byte_offset_low  = (k_base + (lane & 3) * 4) / 2;
    const int byte_offset_high = (k_base + (lane & 3) * 4 + 16) / 2;
    uint32_t r_low  = load_4_nibbles_as_fp8(act_packed, byte_offset_low);
    uint32_t r_high = load_4_nibbles_as_fp8(act_packed, byte_offset_high);
    a_low[0]  = r_low;  a_low[1]  = r_low;
    a_low[2]  = 0;       a_low[3]  = 0;
    a_high[0] = 0;       a_high[1] = 0;
    a_high[2] = r_high; a_high[3] = r_high;
}

__device__ __forceinline__ void fill_b_8rows_two_halves_vec(
    const uint8_t* weight_rows_packed,
    int k_total,
    int k_base,
    int lane,
    uint32_t b_low[2], uint32_t b_high[2]
) {
    const int n_row = lane >> 2;
    const uint8_t* row_ptr = weight_rows_packed + n_row * (k_total / 2);
    const int byte_offset_low  = (k_base + (lane & 3) * 4) / 2;
    const int byte_offset_high = (k_base + (lane & 3) * 4 + 16) / 2;
    b_low[0]  = load_4_nibbles_as_fp8(row_ptr, byte_offset_low);
    b_low[1]  = 0;
    b_high[0] = 0;
    b_high[1] = load_4_nibbles_as_fp8(row_ptr, byte_offset_high);
}

__global__ void sparkfp8_gate_up_kernel(
    const uint8_t* __restrict__ act_packed,
    const float*   __restrict__ act_scale,
    const int32_t* __restrict__ expert_ids,
    const uint8_t* __restrict__ gate_up_packed,
    const float*   __restrict__ gate_up_scale,
    float          gate_up_global_scale,
    int hidden,
    int intermediate,
    int top_k,
    float* __restrict__ gate_up_out
) {
    const int rows_per_tile = 8;
    const int num_row_tiles = (2 * intermediate) / rows_per_tile;
    const int slot = blockIdx.x / num_row_tiles;
    const int row_tile = blockIdx.x % num_row_tiles;
    if (slot >= top_k) return;

    const int expert_id = expert_ids[slot];
    const int row_start = row_tile * rows_per_tile;

    const uint8_t* expert_weight = gate_up_packed +
        (int64_t)expert_id * (2 * intermediate) * (hidden / 2) +
        (int64_t)row_start * (hidden / 2);
    const float* expert_scale = gate_up_scale +
        (int64_t)expert_id * (2 * intermediate) * (hidden / 16) +
        (int64_t)row_start * (hidden / 16);

    const int lane = threadIdx.x;
    const int n_col_a = (lane & 3) * 2 + 0;
    const int n_col_b = (lane & 3) * 2 + 1;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const int num_k32 = hidden / 32;
    const int k_div16 = hidden / 16;

    #pragma unroll 1
    for (int t = 0; t < num_k32; ++t) {
        const int k_base = t * 32;
        const float a_scale_low  = act_scale[k_base / 16];
        const float a_scale_high = act_scale[k_base / 16 + 1];
        const float w_scale_a_low  = expert_scale[n_col_a * k_div16 + k_base / 16];
        const float w_scale_b_low  = expert_scale[n_col_b * k_div16 + k_base / 16];
        const float w_scale_a_high = expert_scale[n_col_a * k_div16 + k_base / 16 + 1];
        const float w_scale_b_high = expert_scale[n_col_b * k_div16 + k_base / 16 + 1];

        uint32_t a_low[4], a_high[4];
        uint32_t b_low[2], b_high[2];
        fill_a_two_halves_vec(act_packed, k_base, lane, a_low, a_high);
        fill_b_8rows_two_halves_vec(expert_weight, hidden, k_base, lane, b_low, b_high);

        float d_low[4], d_high[4];
        fp8_mma_m16n8k32(a_low,  b_low,  d_low);
        fp8_mma_m16n8k32(a_high, b_high, d_high);

        const float scale_a_low  = a_scale_low  * w_scale_a_low  / gate_up_global_scale;
        const float scale_b_low  = a_scale_low  * w_scale_b_low  / gate_up_global_scale;
        const float scale_a_high = a_scale_high * w_scale_a_high / gate_up_global_scale;
        const float scale_b_high = a_scale_high * w_scale_b_high / gate_up_global_scale;

        acc[0] += d_low[0] * scale_a_low + d_high[0] * scale_a_high;
        acc[1] += d_low[1] * scale_b_low + d_high[1] * scale_b_high;
        acc[2] += d_low[2] * scale_a_low + d_high[2] * scale_a_high;
        acc[3] += d_low[3] * scale_b_low + d_high[3] * scale_b_high;
    }

    if (lane < 4) {
        const int64_t base = (int64_t)slot * (2 * intermediate) + row_start;
        gate_up_out[base + lane * 2 + 0] = acc[0];
        gate_up_out[base + lane * 2 + 1] = acc[1];
    }
}

__global__ void sparkfp8_down_kernel(
    const uint8_t* __restrict__ inter_packed,
    const float*   __restrict__ inter_scale,
    const int32_t* __restrict__ expert_ids,
    const uint8_t* __restrict__ down_packed,
    const float*   __restrict__ down_scale,
    float          down_global_scale,
    int hidden,
    int intermediate,
    int top_k,
    float* __restrict__ down_out
) {
    const int rows_per_tile = 8;
    const int num_row_tiles = hidden / rows_per_tile;
    const int slot = blockIdx.x / num_row_tiles;
    const int row_tile = blockIdx.x % num_row_tiles;
    if (slot >= top_k) return;

    const int expert_id = expert_ids[slot];
    const int row_start = row_tile * rows_per_tile;

    const uint8_t* expert_weight = down_packed +
        (int64_t)expert_id * hidden * (intermediate / 2) +
        (int64_t)row_start * (intermediate / 2);
    const float* expert_scale = down_scale +
        (int64_t)expert_id * hidden * (intermediate / 16) +
        (int64_t)row_start * (intermediate / 16);
    const uint8_t* slot_inter = inter_packed + (int64_t)slot * (intermediate / 2);
    const float* slot_inter_scale = inter_scale + (int64_t)slot * (intermediate / 16);

    const int lane = threadIdx.x;
    const int n_col_a = (lane & 3) * 2 + 0;
    const int n_col_b = (lane & 3) * 2 + 1;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const int num_k32 = intermediate / 32;
    const int k_div16 = intermediate / 16;

    #pragma unroll 1
    for (int t = 0; t < num_k32; ++t) {
        const int k_base = t * 32;
        const float a_scale_low  = slot_inter_scale[k_base / 16];
        const float a_scale_high = slot_inter_scale[k_base / 16 + 1];
        const float w_scale_a_low  = expert_scale[n_col_a * k_div16 + k_base / 16];
        const float w_scale_b_low  = expert_scale[n_col_b * k_div16 + k_base / 16];
        const float w_scale_a_high = expert_scale[n_col_a * k_div16 + k_base / 16 + 1];
        const float w_scale_b_high = expert_scale[n_col_b * k_div16 + k_base / 16 + 1];

        uint32_t a_low[4], a_high[4];
        uint32_t b_low[2], b_high[2];
        fill_a_two_halves_vec(slot_inter, k_base, lane, a_low, a_high);
        fill_b_8rows_two_halves_vec(expert_weight, intermediate, k_base, lane, b_low, b_high);

        float d_low[4], d_high[4];
        fp8_mma_m16n8k32(a_low,  b_low,  d_low);
        fp8_mma_m16n8k32(a_high, b_high, d_high);

        const float scale_a_low  = a_scale_low  * w_scale_a_low  / down_global_scale;
        const float scale_b_low  = a_scale_low  * w_scale_b_low  / down_global_scale;
        const float scale_a_high = a_scale_high * w_scale_a_high / down_global_scale;
        const float scale_b_high = a_scale_high * w_scale_b_high / down_global_scale;

        acc[0] += d_low[0] * scale_a_low + d_high[0] * scale_a_high;
        acc[1] += d_low[1] * scale_b_low + d_high[1] * scale_b_high;
        acc[2] += d_low[2] * scale_a_low + d_high[2] * scale_a_high;
        acc[3] += d_low[3] * scale_b_low + d_high[3] * scale_b_high;
    }

    if (lane < 4) {
        const int64_t base = (int64_t)slot * hidden + row_start;
        down_out[base + lane * 2 + 0] = acc[0];
        down_out[base + lane * 2 + 1] = acc[1];
    }
}

}  // namespace

torch::Tensor sparkfp8_gate_up(
    torch::Tensor act_packed, torch::Tensor act_scale,
    torch::Tensor expert_ids,
    torch::Tensor gate_up_packed, torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    int64_t intermediate
) {
    const int hidden = act_packed.numel() * 2;
    const int top_k = expert_ids.numel();
    auto out = torch::zeros({top_k, 2 * (int)intermediate},
                            torch::dtype(torch::kFloat32).device(act_packed.device()));
    const int num_row_tiles = (2 * intermediate) / 8;
    const int total_blocks = top_k * num_row_tiles;
    sparkfp8_gate_up_kernel<<<total_blocks, 32>>>(
        act_packed.contiguous().data_ptr<uint8_t>(),
        act_scale.contiguous().data_ptr<float>(),
        expert_ids.contiguous().data_ptr<int32_t>(),
        gate_up_packed.contiguous().data_ptr<uint8_t>(),
        gate_up_scale.contiguous().data_ptr<float>(),
        gate_up_global_scale.item<float>(),
        hidden, (int)intermediate, top_k,
        out.data_ptr<float>()
    );
    return out;
}

torch::Tensor sparkfp8_down(
    torch::Tensor inter_packed, torch::Tensor inter_scale,
    torch::Tensor expert_ids,
    torch::Tensor down_packed, torch::Tensor down_scale,
    torch::Tensor down_global_scale,
    int64_t hidden
) {
    const int top_k = expert_ids.numel();
    const int intermediate = inter_packed.size(1) * 2;
    auto out = torch::zeros({top_k, (int)hidden},
                            torch::dtype(torch::kFloat32).device(inter_packed.device()));
    const int num_row_tiles = hidden / 8;
    const int total_blocks = top_k * num_row_tiles;
    sparkfp8_down_kernel<<<total_blocks, 32>>>(
        inter_packed.contiguous().data_ptr<uint8_t>(),
        inter_scale.contiguous().data_ptr<float>(),
        expert_ids.contiguous().data_ptr<int32_t>(),
        down_packed.contiguous().data_ptr<uint8_t>(),
        down_scale.contiguous().data_ptr<float>(),
        down_global_scale.item<float>(),
        (int)hidden, intermediate, top_k,
        out.data_ptr<float>()
    );
    return out;
}
"""


def _build_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    from torch.utils.cpp_extension import load

    build_dir = Path(os.environ.get("LYNN_SPARK_FP8_BUILD_DIR", _BUILD_DIR_DEFAULT))
    build_dir.mkdir(parents=True, exist_ok=True)
    cpp_path = build_dir / "spark_fp8_bindings.cpp"
    cu_path = build_dir / "spark_fp8_kernel.cu"
    cpp_path.write_text(_CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(_CUDA_SOURCE, encoding="utf-8")

    arch = os.environ.get("LYNN_NATIVE_CUDA_ARCH", "sm_121a")
    _EXTENSION = load(
        name="lynn_spark_fp8",
        sources=[str(cpp_path), str(cu_path)],
        build_directory=str(build_dir),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", f"-arch={arch}"],
        verbose=False,
    )
    return _EXTENSION


def active_moe_spark_fp8(
    hidden_bf16: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global_scale: torch.Tensor,
) -> torch.Tensor:
    """Spark sm_121 FP8 native active expert MoE forward.

    Args:
      hidden_bf16: [HIDDEN] BF16 or FP16 input activation
      expert_ids: [top_k] int32 (Lynn router output)
      routing_weights: [top_k] float32 (Lynn softmax weights)
      gate_up_packed: [E, 2*INTER, HIDDEN/2] uint8 (Lynn-native E2M1)
      gate_up_scale: [E, 2*INTER, HIDDEN/16] float32 (Lynn per-16 scale)
      gate_up_global_scale: [1] float32
      down_packed: [E, HIDDEN, INTER/2] uint8
      down_scale: [E, HIDDEN, INTER/16] float32
      down_global_scale: [1] float32

    Returns:
      [HIDDEN] BF16 (matches input dtype) — active-MoE output

    Numerical contract: equivalent to scalar reference (E2M1 LUT decode +
    per-16 scale apply + matmul + SiLU + routing) within FP32 accumulation
    tolerance (~1e-6 max_abs_err on real Lynn 27B inputs).
    """
    ext = _build_extension()

    # Quantize activation to E2M1 (matches Lynn's existing quantize_fp4_m1_native)
    from triton_kernels.nvfp4_linear import quantize_fp4_m1_native

    h_flat = hidden_bf16.reshape(-1, hidden_bf16.shape[-1])
    if h_flat.shape[0] != 1:
        raise NotImplementedError("spark_fp8 active MoE currently supports batch=1 decode")

    # Defensive dtype conversion — Lynn's quantize_fp4_m1_native may return
    # tensors with float8_e4m3fn / float4_e2m1fn_x2 storage; kernel needs raw
    # uint8 + float32. Also packed weight tensors stored on the runner may be
    # float8 views; convert with .view(torch.uint8) (NOT .to which changes data).
    act_packed_fp4, act_scale_fp = quantize_fp4_m1_native(h_flat.contiguous())
    act_packed = act_packed_fp4.view(torch.uint8).reshape(-1).contiguous()
    act_scale = act_scale_fp.to(torch.float32).reshape(-1).contiguous()

    expert_ids_i32 = expert_ids.to(torch.int32).contiguous()
    intermediate = gate_up_packed.shape[1] // 2

    gate_up_packed_u8 = gate_up_packed if gate_up_packed.dtype == torch.uint8 else gate_up_packed.view(torch.uint8)
    gate_up_scale_f32 = gate_up_scale.to(torch.float32) if gate_up_scale.dtype != torch.float32 else gate_up_scale
    gate_up_global_f32 = gate_up_global_scale.to(torch.float32) if gate_up_global_scale.dtype != torch.float32 else gate_up_global_scale
    down_packed_u8 = down_packed if down_packed.dtype == torch.uint8 else down_packed.view(torch.uint8)
    down_scale_f32 = down_scale.to(torch.float32) if down_scale.dtype != torch.float32 else down_scale
    down_global_f32 = down_global_scale.to(torch.float32) if down_global_scale.dtype != torch.float32 else down_global_scale

    # Stage 1: gate_up
    gate_up_out = ext.gate_up(
        act_packed, act_scale, expert_ids_i32,
        gate_up_packed_u8, gate_up_scale_f32, gate_up_global_f32,
        intermediate,
    )

    # Stage 2: SiLU * up
    gate = gate_up_out[:, :intermediate]
    up = gate_up_out[:, intermediate:]
    inter = F.silu(gate) * up  # [top_k, intermediate] FP32

    # Stage 3: quantize inter [top_k, INTER] to E2M1 packed + per-16 scale.
    # Use our batched helper since Lynn's quantize_fp4_m1_native only handles [K]/[1,K].
    inter_packed, inter_scale = _quantize_e2m1_batched(inter.contiguous())
    inter_packed = inter_packed.contiguous()
    inter_scale = inter_scale.contiguous()

    hidden_dim = hidden_bf16.shape[-1]
    down_out = ext.down(
        inter_packed, inter_scale, expert_ids_i32,
        down_packed_u8, down_scale_f32, down_global_f32,
        hidden_dim,
    )

    # Stage 4: routing weighted sum
    moe_out = (routing_weights.to(torch.float32).unsqueeze(1) * down_out).sum(dim=0)  # [HIDDEN]

    return moe_out.to(hidden_bf16.dtype).reshape_as(hidden_bf16)


__all__ = ["active_moe_spark_fp8"]
