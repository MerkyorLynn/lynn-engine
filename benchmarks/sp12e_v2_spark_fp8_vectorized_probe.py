#!/usr/bin/env python3
"""SP-12-E-v2: Spark FP8 active-MoE with vectorized loads + register-LUT.

Key changes from SP-12-D / SP-12-E-v1:
  1. uint16 loads for nibble bytes (compiler may already coalesce, but
     explicit form makes it certain)
  2. Register-based E2M1 -> FP8 LUT (avoid __constant__ memory
     serialization for non-uniform-across-thread access)
  3. Keep SP-12-D block layout (32 threads/block, no shared mem) — SP-12-E-v1
     proved multi-warp doesn't help here (parallelism was already saturated)

Target: 175us -> 80-100us gate_up via:
  - Memory: 8 per-byte reads -> 2 per-uint16 reads per K tile per lane = 4x fewer ops
  - LUT: 8 constant-mem lookups -> 8 shift+mask on 4 uint32 registers

Reuses the proven-correct contract from SP-12-D-v1.
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

torch::Tensor sp12e2_gate_up(
    torch::Tensor act_packed,
    torch::Tensor act_scale,
    torch::Tensor expert_ids,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    int64_t intermediate
);

torch::Tensor sp12e2_down(
    torch::Tensor inter_packed,
    torch::Tensor inter_scale,
    torch::Tensor expert_ids,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale,
    int64_t hidden
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gate_up", &sp12e2_gate_up, "SP-12-E-v2 vectorized + register LUT");
  m.def("down", &sp12e2_down, "SP-12-E-v2 down vectorized + register LUT");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

// E2M1 nibble -> FP8 E4M3 byte LUT packed into 4 uint32 registers:
//   LUT_REG_0 (bytes 0..3):   mag 0,1,2,3 positive   = 0x00, 0x30, 0x38, 0x3C
//   LUT_REG_1 (bytes 4..7):   mag 4,5,6,7 positive   = 0x40, 0x44, 0x48, 0x4C
//   LUT_REG_2 (bytes 8..11):  mag 0,1,2,3 negative   = 0x80, 0xB0, 0xB8, 0xBC
//   LUT_REG_3 (bytes 12..15): mag 4,5,6,7 negative   = 0xC0, 0xC4, 0xC8, 0xCC
//
// Note: LUT_REG_x packs 4 bytes in little-endian uint32:
//   byte_idx 0 -> bits 0..7, byte_idx 1 -> bits 8..15, etc.
//
// LUT_REG_0 = (0x00) | (0x30 << 8) | (0x38 << 16) | (0x3C << 24) = 0x3C383000
// LUT_REG_1 = (0x40) | (0x44 << 8) | (0x48 << 16) | (0x4C << 24) = 0x4C484440
// LUT_REG_2 = (0x80) | (0xB0 << 8) | (0xB8 << 16) | (0xBC << 24) = 0xBCB8B080
// LUT_REG_3 = (0xC0) | (0xC4 << 8) | (0xC8 << 16) | (0xCC << 24) = 0xCCC8C4C0

constexpr uint32_t LUT_REG_0 = 0x3C383000;
constexpr uint32_t LUT_REG_1 = 0x4C484440;
constexpr uint32_t LUT_REG_2 = 0xBCB8B080;
constexpr uint32_t LUT_REG_3 = 0xCCC8C4C0;

__device__ __forceinline__ uint8_t lut_nibble_to_fp8(uint8_t nibble) {
    // Use byte-index 0..15 to pick from 4 packed LUT registers
    // byte_idx = nibble (since LUT is packed in the same order as nibble values)
    uint32_t reg;
    uint8_t bi = nibble & 0x0F;
    if (bi < 4)        reg = LUT_REG_0;
    else if (bi < 8)   reg = LUT_REG_1;
    else if (bi < 12)  reg = LUT_REG_2;
    else               reg = LUT_REG_3;
    uint8_t shift = (bi & 3) * 8;
    return (uint8_t)((reg >> shift) & 0xFF);
}

// Vectorized: load 16 bits = 2 bytes = 4 nibbles, return as a uint32 of 4 FP8 bytes
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
    // Each lane reads 4 nibbles at k_base + (lane%4)*4 (for low half)
    // and 4 nibbles at k_base + (lane%4)*4 + 16 (for high half)
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

__global__ void sp12e2_gate_up_kernel(
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

__global__ void sp12e2_down_kernel(
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

torch::Tensor sp12e2_gate_up(
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
    sp12e2_gate_up_kernel<<<total_blocks, 32>>>(
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

torch::Tensor sp12e2_down(
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
    sp12e2_down_kernel<<<total_blocks, 32>>>(
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


# Reuse the driver from SP-12-D
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
        packed = packed.squeeze(0); scale = scale.squeeze(0)
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
    (build_root / "bind.cpp").write_text(CPP_SOURCE)
    (build_root / "kern.cu").write_text(CUDA_SOURCE)
    return load(
        name="lynn_sp12e2_vec",
        sources=[str(build_root / "bind.cpp"), str(build_root / "kern.cu")],
        build_directory=str(build_root),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_121a"],
        verbose=verbose,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_sp12e2_build")
    ap.add_argument("--seed", type=int, default=20260516)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cap = torch.cuda.get_device_capability(0)
    print(f"[sp12e2] device: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")

    print(f"[sp12e2] building...")
    t0 = time.time()
    module = _build_module(Path(args.build_dir), args.verbose)
    print(f"[sp12e2] build OK in {time.time() - t0:.1f}s")

    torch.manual_seed(args.seed)
    H, I, E, TK = 2048, 512, 256, 8

    act_fp32 = torch.randn(H, dtype=torch.float32, device="cuda") * 0.5
    act_packed, act_scale = quantize_to_e2m1(act_fp32)
    expert_ids = torch.randint(0, E, (TK,), dtype=torch.int32, device="cuda")
    routing = F.softmax(torch.randn(TK, dtype=torch.float32, device="cuda"), dim=0)

    print(f"[sp12e2] quantizing 256 expert weights...")
    gate_up_fp32 = torch.randn(E, 2*I, H, dtype=torch.float32, device="cuda") * 0.05
    gate_up_packed = torch.empty(E, 2*I, H//2, dtype=torch.uint8, device="cuda")
    gate_up_scale = torch.empty(E, 2*I, H//16, dtype=torch.float32, device="cuda")
    for e in range(E):
        p, s = quantize_to_e2m1(gate_up_fp32[e])
        gate_up_packed[e] = p
        gate_up_scale[e] = s
    gate_up_global = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    down_fp32 = torch.randn(E, H, I, dtype=torch.float32, device="cuda") * 0.05
    down_packed = torch.empty(E, H, I//2, dtype=torch.uint8, device="cuda")
    down_scale = torch.empty(E, H, I//16, dtype=torch.float32, device="cuda")
    for e in range(E):
        p, s = quantize_to_e2m1(down_fp32[e])
        down_packed[e] = p
        down_scale[e] = s
    down_global = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    print(f"[sp12e2] gate_up...")
    gate_up_out = module.gate_up(act_packed, act_scale, expert_ids,
                                 gate_up_packed, gate_up_scale, gate_up_global, I)
    torch.cuda.synchronize()

    print(f"[sp12e2] gate_up reference...")
    act_dec = decode_packed(act_packed, act_scale)
    gate_up_ref = torch.zeros(TK, 2*I, device="cuda")
    for slot in range(TK):
        e = int(expert_ids[slot].item())
        w = decode_packed(gate_up_packed[e], gate_up_scale[e])
        gate_up_ref[slot] = w @ act_dec
    gate_up_abs = float((gate_up_out - gate_up_ref).abs().max().item())
    gate_up_pass = gate_up_abs < 1e-3 * float(gate_up_ref.abs().max().item())
    print(f"[sp12e2] gate_up max_abs_err: {gate_up_abs:.3e}  PASS: {gate_up_pass}")

    gate = gate_up_out[:, :I]
    up = gate_up_out[:, I:]
    inter = F.silu(gate) * up
    inter_packed, inter_scale = quantize_to_e2m1(inter)

    print(f"[sp12e2] down...")
    down_out = module.down(inter_packed, inter_scale, expert_ids,
                           down_packed, down_scale, down_global, H)
    torch.cuda.synchronize()
    print(f"[sp12e2] down reference...")
    inter_dec = decode_packed(inter_packed, inter_scale)
    down_ref = torch.zeros(TK, H, device="cuda")
    for slot in range(TK):
        e = int(expert_ids[slot].item())
        w = decode_packed(down_packed[e], down_scale[e])
        down_ref[slot] = w @ inter_dec[slot]
    down_abs = float((down_out - down_ref).abs().max().item())
    down_pass = down_abs < 1e-3 * float(down_ref.abs().max().item())
    print(f"[sp12e2] down max_abs_err: {down_abs:.3e}  PASS: {down_pass}")

    # Timing
    print(f"[sp12e2] timing...")
    for _ in range(5):
        module.gate_up(act_packed, act_scale, expert_ids, gate_up_packed, gate_up_scale, gate_up_global, I)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e_evt = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        module.gate_up(act_packed, act_scale, expert_ids, gate_up_packed, gate_up_scale, gate_up_global, I)
    e_evt.record()
    torch.cuda.synchronize()
    gu_us = float(s.elapsed_time(e_evt) / 50 * 1000.0)

    for _ in range(5):
        module.down(inter_packed, inter_scale, expert_ids, down_packed, down_scale, down_global, H)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e_evt = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        module.down(inter_packed, inter_scale, expert_ids, down_packed, down_scale, down_global, H)
    e_evt.record()
    torch.cuda.synchronize()
    dn_us = float(s.elapsed_time(e_evt) / 50 * 1000.0)

    sp12d_total = 175.24 + 88.31
    new_total = gu_us + dn_us
    speedup = sp12d_total / new_total
    print(f"[sp12e2] gate_up: {gu_us:.2f} us  down: {dn_us:.2f} us  total: {new_total:.2f} us")
    print(f"[sp12e2] vs SP-12-D {sp12d_total:.2f} us  speedup: {speedup:.2f}x")

    summary = {
        "type": "sp12e_v2_vectorized_register_lut",
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "gate_up_us": gu_us, "down_us": dn_us, "total_us": new_total,
        "gate_up_max_abs_err": gate_up_abs,
        "down_max_abs_err": down_abs,
        "all_pass": gate_up_pass and down_pass,
        "speedup_vs_sp12d": speedup,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[sp12e2] === SUMMARY === total={new_total:.2f}us  speedup={speedup:.2f}x  PASS={summary['all_pass']}")
    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
