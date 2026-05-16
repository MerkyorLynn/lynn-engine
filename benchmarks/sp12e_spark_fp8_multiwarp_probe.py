#!/usr/bin/env python3
"""SP-12-E-v1: Spark FP8 active-MoE with multi-warp blocks + shared-mem cache.

Optimization v1 over SP-12-D (correctness-validated baseline):

  Block design changes from SP-12-D (1 warp/block) to SP-12-E-v1 (4 warps/block):
    - 128 threads/block, 4 warps
    - Each warp handles its own row_tile of 8 (4 row tiles per block = 32 output rows)
    - Shared mem caches activation (1024 bytes packed + 128 floats scale)
    - All 4 warps reuse the SAME shared activation across K loop

  Grid changes:
    Gate_up: 8 experts × (2*INTER/32) = 8 × 32 = 256 blocks (was 1024)
    Down:    8 experts × (HIDDEN/32) = 8 × 64 = 512 blocks (was 2048)

  Expected gain: ~2-3× over SP-12-D-v1 from:
    - 4x more tensor-core throughput per block (warps run concurrently)
    - Activation HBM reads reduced 4x (shared in block)
    - Better SM occupancy (each block uses 4 warp slots cooperatively)

Promotion gate: numerical parity vs SP-12-D within 1e-5 + per-call latency
ratio < 0.5 (i.e. 2x faster).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor sparkfp8e_gate_up(
    torch::Tensor act_packed,
    torch::Tensor act_scale,
    torch::Tensor expert_ids,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    int64_t intermediate
);

torch::Tensor sparkfp8e_down(
    torch::Tensor inter_packed,
    torch::Tensor inter_scale,
    torch::Tensor expert_ids,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale,
    int64_t hidden
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gate_up", &sparkfp8e_gate_up, "SP-12-E-v1 gate_up multiwarp+shared");
  m.def("down", &sparkfp8e_down, "SP-12-E-v1 down multiwarp+shared");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__constant__ uint8_t E2M1_TO_E4M3_LUT[16] = {
    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
    0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC
};

__device__ __forceinline__ uint8_t get_nibble(const uint8_t* packed, int elem) {
    const uint8_t byte = packed[elem >> 1];
    return (elem & 1) == 0 ? (byte & 0x0Fu) : ((byte >> 4) & 0x0Fu);
}

__device__ __forceinline__ uint8_t e2m1_to_e4m3(uint8_t nibble) {
    return E2M1_TO_E4M3_LUT[nibble & 0x0F];
}

__device__ __forceinline__ uint32_t pack4_fp8(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
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

__device__ __forceinline__ void fill_a_two_halves_from_shm(
    const uint8_t* act_packed_shm,  // shared mem [K/2]
    int k_base, int lane,
    uint32_t a_low[4], uint32_t a_high[4]
) {
    const int k0  = k_base + (lane & 3) * 4;
    const int k16 = k0 + 16;
    uint8_t b0 = e2m1_to_e4m3(get_nibble(act_packed_shm, k0 + 0));
    uint8_t b1 = e2m1_to_e4m3(get_nibble(act_packed_shm, k0 + 1));
    uint8_t b2 = e2m1_to_e4m3(get_nibble(act_packed_shm, k0 + 2));
    uint8_t b3 = e2m1_to_e4m3(get_nibble(act_packed_shm, k0 + 3));
    uint8_t h0 = e2m1_to_e4m3(get_nibble(act_packed_shm, k16 + 0));
    uint8_t h1 = e2m1_to_e4m3(get_nibble(act_packed_shm, k16 + 1));
    uint8_t h2 = e2m1_to_e4m3(get_nibble(act_packed_shm, k16 + 2));
    uint8_t h3 = e2m1_to_e4m3(get_nibble(act_packed_shm, k16 + 3));
    uint32_t r_low  = pack4_fp8(b0, b1, b2, b3);
    uint32_t r_high = pack4_fp8(h0, h1, h2, h3);
    a_low[0]  = r_low;  a_low[1]  = r_low;
    a_low[2]  = 0;       a_low[3]  = 0;
    a_high[0] = 0;       a_high[1] = 0;
    a_high[2] = r_high; a_high[3] = r_high;
}

__device__ __forceinline__ void fill_b_8rows_two_halves(
    const uint8_t* weight_rows_packed,
    int k_total,
    int k_base,
    int lane,
    uint32_t b_low[2], uint32_t b_high[2]
) {
    const int n_row = lane >> 2;
    const uint8_t* row_ptr = weight_rows_packed + n_row * (k_total / 2);
    const int k0  = k_base + (lane & 3) * 4;
    const int k16 = k0 + 16;
    uint8_t b0 = e2m1_to_e4m3(get_nibble(row_ptr, k0 + 0));
    uint8_t b1 = e2m1_to_e4m3(get_nibble(row_ptr, k0 + 1));
    uint8_t b2 = e2m1_to_e4m3(get_nibble(row_ptr, k0 + 2));
    uint8_t b3 = e2m1_to_e4m3(get_nibble(row_ptr, k0 + 3));
    uint8_t h0 = e2m1_to_e4m3(get_nibble(row_ptr, k16 + 0));
    uint8_t h1 = e2m1_to_e4m3(get_nibble(row_ptr, k16 + 1));
    uint8_t h2 = e2m1_to_e4m3(get_nibble(row_ptr, k16 + 2));
    uint8_t h3 = e2m1_to_e4m3(get_nibble(row_ptr, k16 + 3));
    b_low[0]  = pack4_fp8(b0, b1, b2, b3);
    b_low[1]  = 0;
    b_high[0] = 0;
    b_high[1] = pack4_fp8(h0, h1, h2, h3);
}

// SP-12-E gate_up: 4 warps per block (128 threads), shared-mem activation cache.
// Each warp handles its own row_tile of 8 rows. Block handles 4 × 8 = 32 rows.
// Grid: top_k × (2*INTER / 32) = 8 × 32 = 256 blocks (for INTER=512).
__global__ void sparkfp8e_gate_up_kernel(
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
    constexpr int ROWS_PER_WARP = 8;
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int ROWS_PER_BLOCK = ROWS_PER_WARP * WARPS_PER_BLOCK;  // 32
    const int num_row_blocks = (2 * intermediate) / ROWS_PER_BLOCK;
    const int slot = blockIdx.x / num_row_blocks;
    const int row_block = blockIdx.x % num_row_blocks;
    if (slot >= top_k) return;

    const int expert_id = expert_ids[slot];
    const int row_block_start = row_block * ROWS_PER_BLOCK;

    const int warp_idx = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int my_row_start = row_block_start + warp_idx * ROWS_PER_WARP;

    // Shared mem for activation
    extern __shared__ uint8_t shm[];
    uint8_t* act_packed_shm = shm;                                              // [hidden/2]
    float*   act_scale_shm  = reinterpret_cast<float*>(shm + (hidden / 2));     // [hidden/16]

    // Cooperative load of activation into shared mem (128 threads / 8 bytes = 16 iters for 1024 bytes)
    const int packed_bytes = hidden / 2;
    for (int i = threadIdx.x; i < packed_bytes; i += blockDim.x) {
        act_packed_shm[i] = act_packed[i];
    }
    const int scale_count = hidden / 16;
    for (int i = threadIdx.x; i < scale_count; i += blockDim.x) {
        act_scale_shm[i] = act_scale[i];
    }
    __syncthreads();

    // Each warp's own row weights
    const uint8_t* expert_weight = gate_up_packed +
        (int64_t)expert_id * (2 * intermediate) * (hidden / 2) +
        (int64_t)my_row_start * (hidden / 2);
    const float* expert_scale = gate_up_scale +
        (int64_t)expert_id * (2 * intermediate) * (hidden / 16) +
        (int64_t)my_row_start * (hidden / 16);

    const int n_col_a = (lane & 3) * 2 + 0;
    const int n_col_b = (lane & 3) * 2 + 1;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const int num_k32 = hidden / 32;
    const int k_div16 = hidden / 16;

    #pragma unroll 1
    for (int t = 0; t < num_k32; ++t) {
        const int k_base = t * 32;
        const float a_scale_low  = act_scale_shm[k_base / 16];
        const float a_scale_high = act_scale_shm[k_base / 16 + 1];
        const float w_scale_a_low  = expert_scale[n_col_a * k_div16 + k_base / 16];
        const float w_scale_b_low  = expert_scale[n_col_b * k_div16 + k_base / 16];
        const float w_scale_a_high = expert_scale[n_col_a * k_div16 + k_base / 16 + 1];
        const float w_scale_b_high = expert_scale[n_col_b * k_div16 + k_base / 16 + 1];

        uint32_t a_low[4], a_high[4];
        uint32_t b_low[2], b_high[2];
        fill_a_two_halves_from_shm(act_packed_shm, k_base, lane, a_low, a_high);
        fill_b_8rows_two_halves(expert_weight, hidden, k_base, lane, b_low, b_high);

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
        const int64_t base = (int64_t)slot * (2 * intermediate) + my_row_start;
        gate_up_out[base + lane * 2 + 0] = acc[0];
        gate_up_out[base + lane * 2 + 1] = acc[1];
    }
}

__global__ void sparkfp8e_down_kernel(
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
    constexpr int ROWS_PER_WARP = 8;
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int ROWS_PER_BLOCK = ROWS_PER_WARP * WARPS_PER_BLOCK;  // 32
    const int num_row_blocks = hidden / ROWS_PER_BLOCK;
    const int slot = blockIdx.x / num_row_blocks;
    const int row_block = blockIdx.x % num_row_blocks;
    if (slot >= top_k) return;

    const int expert_id = expert_ids[slot];
    const int row_block_start = row_block * ROWS_PER_BLOCK;

    const int warp_idx = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int my_row_start = row_block_start + warp_idx * ROWS_PER_WARP;

    // Shared mem for THIS slot's inter (input to down kernel)
    extern __shared__ uint8_t shm[];
    uint8_t* slot_inter_shm = shm;                                                          // [intermediate/2]
    float*   slot_scale_shm = reinterpret_cast<float*>(shm + (intermediate / 2));            // [intermediate/16]

    const int packed_bytes = intermediate / 2;
    const uint8_t* slot_inter = inter_packed + (int64_t)slot * (intermediate / 2);
    const float* slot_inter_scale_g = inter_scale + (int64_t)slot * (intermediate / 16);
    for (int i = threadIdx.x; i < packed_bytes; i += blockDim.x) {
        slot_inter_shm[i] = slot_inter[i];
    }
    const int scale_count = intermediate / 16;
    for (int i = threadIdx.x; i < scale_count; i += blockDim.x) {
        slot_scale_shm[i] = slot_inter_scale_g[i];
    }
    __syncthreads();

    const uint8_t* expert_weight = down_packed +
        (int64_t)expert_id * hidden * (intermediate / 2) +
        (int64_t)my_row_start * (intermediate / 2);
    const float* expert_scale = down_scale +
        (int64_t)expert_id * hidden * (intermediate / 16) +
        (int64_t)my_row_start * (intermediate / 16);

    const int n_col_a = (lane & 3) * 2 + 0;
    const int n_col_b = (lane & 3) * 2 + 1;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const int num_k32 = intermediate / 32;
    const int k_div16 = intermediate / 16;

    #pragma unroll 1
    for (int t = 0; t < num_k32; ++t) {
        const int k_base = t * 32;
        const float a_scale_low  = slot_scale_shm[k_base / 16];
        const float a_scale_high = slot_scale_shm[k_base / 16 + 1];
        const float w_scale_a_low  = expert_scale[n_col_a * k_div16 + k_base / 16];
        const float w_scale_b_low  = expert_scale[n_col_b * k_div16 + k_base / 16];
        const float w_scale_a_high = expert_scale[n_col_a * k_div16 + k_base / 16 + 1];
        const float w_scale_b_high = expert_scale[n_col_b * k_div16 + k_base / 16 + 1];

        uint32_t a_low[4], a_high[4];
        uint32_t b_low[2], b_high[2];
        fill_a_two_halves_from_shm(slot_inter_shm, k_base, lane, a_low, a_high);
        fill_b_8rows_two_halves(expert_weight, intermediate, k_base, lane, b_low, b_high);

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
        const int64_t base = (int64_t)slot * hidden + my_row_start;
        down_out[base + lane * 2 + 0] = acc[0];
        down_out[base + lane * 2 + 1] = acc[1];
    }
}

}  // namespace

torch::Tensor sparkfp8e_gate_up(
    torch::Tensor act_packed,
    torch::Tensor act_scale,
    torch::Tensor expert_ids,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    int64_t intermediate
) {
    const int hidden = act_packed.numel() * 2;
    const int top_k = expert_ids.numel();
    auto out = torch::zeros({top_k, 2 * (int)intermediate},
                            torch::dtype(torch::kFloat32).device(act_packed.device()));
    const int rows_per_block = 32;
    const int num_row_blocks = (2 * intermediate) / rows_per_block;
    const int total_blocks = top_k * num_row_blocks;
    const size_t shm_size = (hidden / 2) + (hidden / 16) * sizeof(float);
    sparkfp8e_gate_up_kernel<<<total_blocks, 128, shm_size>>>(
        act_packed.contiguous().data_ptr<uint8_t>(),
        act_scale.contiguous().data_ptr<float>(),
        expert_ids.contiguous().data_ptr<int32_t>(),
        gate_up_packed.contiguous().data_ptr<uint8_t>(),
        gate_up_scale.contiguous().data_ptr<float>(),
        gate_up_global_scale.item<float>(),
        hidden,
        (int)intermediate,
        top_k,
        out.data_ptr<float>()
    );
    return out;
}

torch::Tensor sparkfp8e_down(
    torch::Tensor inter_packed,
    torch::Tensor inter_scale,
    torch::Tensor expert_ids,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale,
    int64_t hidden
) {
    const int top_k = expert_ids.numel();
    const int intermediate = inter_packed.size(1) * 2;
    auto out = torch::zeros({top_k, (int)hidden},
                            torch::dtype(torch::kFloat32).device(inter_packed.device()));
    const int rows_per_block = 32;
    const int num_row_blocks = hidden / rows_per_block;
    const int total_blocks = top_k * num_row_blocks;
    const size_t shm_size = (intermediate / 2) + (intermediate / 16) * sizeof(float);
    sparkfp8e_down_kernel<<<total_blocks, 128, shm_size>>>(
        inter_packed.contiguous().data_ptr<uint8_t>(),
        inter_scale.contiguous().data_ptr<float>(),
        expert_ids.contiguous().data_ptr<int32_t>(),
        down_packed.contiguous().data_ptr<uint8_t>(),
        down_scale.contiguous().data_ptr<float>(),
        down_global_scale.item<float>(),
        (int)hidden,
        intermediate,
        top_k,
        out.data_ptr<float>()
    );
    return out;
}
"""


E2M1_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def quantize_to_e2m1(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = x.dim() == 1
    if flat:
        x = x.unsqueeze(0)
    m, k = x.shape
    table = E2M1_TABLE.to(x.device)
    xg = x.reshape(m, k // 16, 16)
    abs_max = xg.abs().amax(dim=-1)
    scale = (abs_max / 6.0).clamp_min(1e-8)
    normalized = (xg.abs() / scale.unsqueeze(-1)).clamp(0, 6.0)
    mag = torch.argmin((normalized.unsqueeze(-1) - table.view(1, 1, 1, -1)).abs(), dim=-1)
    sign = (xg < 0).to(torch.uint8) * 8
    codes = (mag.to(torch.uint8) | sign).reshape(m, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4))
    if flat:
        packed = packed.squeeze(0)
        scale = scale.squeeze(0)
    return packed.contiguous(), scale.contiguous()


def decode_packed(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    codes_low = packed & 0x0F
    codes_high = (packed >> 4) & 0x0F
    codes = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.uint8, device=packed.device)
    codes[..., 0::2] = codes_low
    codes[..., 1::2] = codes_high
    mag = (codes & 0x07).long()
    sign = ((codes & 0x08) != 0).float() * -2 + 1
    table = E2M1_TABLE.to(packed.device)
    values = table[mag] * sign
    K = values.shape[-1]
    values_grouped = values.reshape(*values.shape[:-1], K // 16, 16)
    scaled = values_grouped * scale.unsqueeze(-1)
    return scaled.reshape(*values.shape[:-1], K)


def _build_module(build_root: Path, verbose: bool):
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True, exist_ok=True)
    cpp_path = build_root / "sp12e_bindings.cpp"
    cu_path = build_root / "sp12e_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")
    return load(
        name="lynn_sp12e_spark_fp8_multiwarp",
        sources=[str(cpp_path), str(cu_path)],
        build_directory=str(build_root),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_121a"],
        verbose=verbose,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_sp12e_build")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--intermediate", type=int, default=512)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260516)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cap = torch.cuda.get_device_capability(0)
    print(f"[sp12e] device: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
    print(f"[sp12e] hidden={args.hidden} intermediate={args.intermediate} top_k={args.top_k}")

    print(f"[sp12e] building...")
    t0 = time.time()
    module = _build_module(Path(args.build_dir), args.verbose)
    print(f"[sp12e] build OK in {time.time() - t0:.1f}s")

    torch.manual_seed(args.seed)
    H, I, E, TK = args.hidden, args.intermediate, args.num_experts, args.top_k

    act_fp32 = torch.randn(H, dtype=torch.float32, device="cuda") * 0.5
    act_packed, act_scale = quantize_to_e2m1(act_fp32)

    expert_ids = torch.randint(0, E, (TK,), dtype=torch.int32, device="cuda")
    routing_weights = F.softmax(torch.randn(TK, dtype=torch.float32, device="cuda"), dim=0)

    print(f"[sp12e] quantizing {E} expert weights (this takes ~30s)...")
    gate_up_fp32 = torch.randn(E, 2 * I, H, dtype=torch.float32, device="cuda") * 0.05
    gate_up_packed = torch.empty(E, 2 * I, H // 2, dtype=torch.uint8, device="cuda")
    gate_up_scale = torch.empty(E, 2 * I, H // 16, dtype=torch.float32, device="cuda")
    for e in range(E):
        p, s = quantize_to_e2m1(gate_up_fp32[e])
        gate_up_packed[e] = p
        gate_up_scale[e] = s
    gate_up_global = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    down_fp32 = torch.randn(E, H, I, dtype=torch.float32, device="cuda") * 0.05
    down_packed = torch.empty(E, H, I // 2, dtype=torch.uint8, device="cuda")
    down_scale = torch.empty(E, H, I // 16, dtype=torch.float32, device="cuda")
    for e in range(E):
        p, s = quantize_to_e2m1(down_fp32[e])
        down_packed[e] = p
        down_scale[e] = s
    down_global = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    print(f"[sp12e] gate_up SP-12-E-v1 (multi-warp + shared mem)...")
    gate_up_out = module.gate_up(act_packed, act_scale, expert_ids,
                                 gate_up_packed, gate_up_scale, gate_up_global, I)
    torch.cuda.synchronize()

    # Reference
    print(f"[sp12e] computing gate_up reference...")
    act_decoded = decode_packed(act_packed, act_scale)
    gate_up_ref = torch.zeros(TK, 2 * I, device="cuda")
    for slot in range(TK):
        e = int(expert_ids[slot].item())
        w = decode_packed(gate_up_packed[e], gate_up_scale[e])
        gate_up_ref[slot] = w @ act_decoded

    gate_up_diff = (gate_up_out - gate_up_ref).abs()
    gate_up_max_abs = float(gate_up_diff.max().item())
    gate_up_rel = float((gate_up_out - gate_up_ref).norm() / gate_up_ref.norm().clamp_min(1e-9))
    print(f"[sp12e] gate_up max_abs_err: {gate_up_max_abs:.6e}  rel_err: {gate_up_rel:.6e}")
    gate_up_pass = gate_up_max_abs < 1e-3 * float(gate_up_ref.abs().max().item())
    print(f"[sp12e] gate_up PASS: {gate_up_pass}")

    # SiLU + multiply
    gate = gate_up_out[:, :I]
    up = gate_up_out[:, I:]
    inter = F.silu(gate) * up
    inter_packed, inter_scale = quantize_to_e2m1(inter)

    print(f"[sp12e] down SP-12-E-v1...")
    down_out = module.down(inter_packed, inter_scale, expert_ids,
                           down_packed, down_scale, down_global, H)
    torch.cuda.synchronize()

    # Reference for down
    print(f"[sp12e] computing down reference...")
    inter_decoded = decode_packed(inter_packed, inter_scale)
    down_ref = torch.zeros(TK, H, device="cuda")
    for slot in range(TK):
        e = int(expert_ids[slot].item())
        w = decode_packed(down_packed[e], down_scale[e])
        down_ref[slot] = w @ inter_decoded[slot]

    down_diff = (down_out - down_ref).abs()
    down_max_abs = float(down_diff.max().item())
    down_rel = float((down_out - down_ref).norm() / down_ref.norm().clamp_min(1e-9))
    print(f"[sp12e] down max_abs_err: {down_max_abs:.6e}  rel_err: {down_rel:.6e}")
    down_pass = down_max_abs < 1e-3 * float(down_ref.abs().max().item())
    print(f"[sp12e] down PASS: {down_pass}")

    # Timing
    print(f"[sp12e] timing gate_up (50 iters)...")
    for _ in range(5): module.gate_up(act_packed, act_scale, expert_ids, gate_up_packed, gate_up_scale, gate_up_global, I)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e_evt = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        module.gate_up(act_packed, act_scale, expert_ids, gate_up_packed, gate_up_scale, gate_up_global, I)
    e_evt.record()
    torch.cuda.synchronize()
    gate_up_us = float(s.elapsed_time(e_evt) / 50 * 1000.0)

    print(f"[sp12e] timing down (50 iters)...")
    for _ in range(5): module.down(inter_packed, inter_scale, expert_ids, down_packed, down_scale, down_global, H)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e_evt = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        module.down(inter_packed, inter_scale, expert_ids, down_packed, down_scale, down_global, H)
    e_evt.record()
    torch.cuda.synchronize()
    down_us = float(s.elapsed_time(e_evt) / 50 * 1000.0)

    sp12d_gate_up_us = 175.24
    sp12d_down_us = 88.31
    speedup_gate_up = sp12d_gate_up_us / gate_up_us
    speedup_down = sp12d_down_us / down_us
    speedup_total = (sp12d_gate_up_us + sp12d_down_us) / (gate_up_us + down_us)

    print(f"[sp12e] gate_up: {gate_up_us:.2f} us  (vs SP-12-D {sp12d_gate_up_us:.2f} us = {speedup_gate_up:.2f}x speedup)")
    print(f"[sp12e] down:    {down_us:.2f} us  (vs SP-12-D {sp12d_down_us:.2f} us = {speedup_down:.2f}x speedup)")
    print(f"[sp12e] total:   {gate_up_us + down_us:.2f} us  ({speedup_total:.2f}x speedup over SP-12-D)")

    summary = {
        "type": "sp12e_v1_multiwarp_sharedmem",
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "gate_up": {"us": gate_up_us, "max_abs_err": gate_up_max_abs, "pass": gate_up_pass},
        "down": {"us": down_us, "max_abs_err": down_max_abs, "pass": down_pass},
        "speedup_vs_sp12d": {"gate_up": speedup_gate_up, "down": speedup_down, "total": speedup_total},
        "total_us": gate_up_us + down_us,
        "all_pass": gate_up_pass and down_pass,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[sp12e] === SUMMARY === total={gate_up_us + down_us:.2f} us  speedup={speedup_total:.2f}x  ALL PASS={summary['all_pass']}")
    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
