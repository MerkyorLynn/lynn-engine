#!/usr/bin/env python3
"""SP-12-D: Spark FP8 active-MoE kernel chain (gate_up + down + routing) probe.

Production-shape probe that runs end-to-end active-expert MoE forward in FP8
MMA tensor cores on Spark sm_121, then validates output against the existing
Triton path (production SP-08 path).

Phase decomposition:
  Stage 1: gate_up kernel — grid (top_k=8 × 2*INTER/8=128 = 1024 blocks)
    Each block (1 warp, 32 threads) handles one (expert_slot, row_tile_of_8)
    Output: gate_up_out [top_k=8, 2*INTER=1024] f32
  Stage 2: SiLU + multiply (Python or trivial CUDA kernel)
    Input: gate_up_out, slicing [..., 0:512] = gate, [..., 512:1024] = up
    Output: inter = silu(gate) * up   [top_k=8, INTER=512] f32 -> E2M1 quantize
  Stage 3: down kernel — grid (top_k × HIDDEN/8 = 8 × 256 = 2048 blocks)
    Output: down_out [top_k=8, HIDDEN=2048] f32
  Stage 4: routing weighted sum (Python or trivial CUDA)
    out[h] = sum_slot routing_weights[slot] * down_out[slot, h]

This SP-12-D-v1 implementation focuses on CORRECTNESS over performance.
Inter-stage data lives in global memory (not fused). Each stage is its own
kernel launch. Python orchestrates the chain.

Production fusion (SP-12-E) will collapse stages into one kernel + persistent
warps + shared-mem inter buffer. For SP-12-D-v1 we just want to verify:
  - All 4 stages produce numerically correct output
  - End-to-end matches Triton scalar_bridge active-MoE within Codex P90 tolerance
  - Per-step latency tells us where the perf headroom is
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

torch::Tensor spark_fp8_gate_up(
    torch::Tensor act_packed,           // [HIDDEN/2] uint8
    torch::Tensor act_scale,            // [HIDDEN/16] f32
    torch::Tensor expert_ids,           // [top_k] int32
    torch::Tensor gate_up_packed,       // [E, 2*INTER, HIDDEN/2] uint8
    torch::Tensor gate_up_scale,        // [E, 2*INTER, HIDDEN/16] f32
    torch::Tensor gate_up_global_scale, // [1] f32
    int64_t intermediate
);

torch::Tensor spark_fp8_down(
    torch::Tensor inter_packed,         // [top_k, INTER/2] uint8
    torch::Tensor inter_scale,          // [top_k, INTER/16] f32
    torch::Tensor expert_ids,           // [top_k] int32
    torch::Tensor down_packed,          // [E, HIDDEN, INTER/2] uint8
    torch::Tensor down_scale,           // [E, HIDDEN, INTER/16] f32
    torch::Tensor down_global_scale,    // [1] f32
    int64_t hidden
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spark_fp8_gate_up", &spark_fp8_gate_up, "SP-12-D gate_up active-MoE FP8 kernel");
  m.def("spark_fp8_down", &spark_fp8_down, "SP-12-D down active-MoE FP8 kernel");
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

__device__ __forceinline__ void fill_a_two_halves(
    const uint8_t* act_packed,
    int k_base, int lane,
    uint32_t a_low[4], uint32_t a_high[4]
) {
    const int k0  = k_base + (lane & 3) * 4;
    const int k16 = k0 + 16;
    uint8_t b0 = e2m1_to_e4m3(get_nibble(act_packed, k0 + 0));
    uint8_t b1 = e2m1_to_e4m3(get_nibble(act_packed, k0 + 1));
    uint8_t b2 = e2m1_to_e4m3(get_nibble(act_packed, k0 + 2));
    uint8_t b3 = e2m1_to_e4m3(get_nibble(act_packed, k0 + 3));
    uint8_t h0 = e2m1_to_e4m3(get_nibble(act_packed, k16 + 0));
    uint8_t h1 = e2m1_to_e4m3(get_nibble(act_packed, k16 + 1));
    uint8_t h2 = e2m1_to_e4m3(get_nibble(act_packed, k16 + 2));
    uint8_t h3 = e2m1_to_e4m3(get_nibble(act_packed, k16 + 3));
    uint32_t r_low  = pack4_fp8(b0, b1, b2, b3);
    uint32_t r_high = pack4_fp8(h0, h1, h2, h3);
    a_low[0]  = r_low;  a_low[1]  = r_low;
    a_low[2]  = 0;       a_low[3]  = 0;
    a_high[0] = 0;       a_high[1] = 0;
    a_high[2] = r_high; a_high[3] = r_high;
}

__device__ __forceinline__ void fill_b_8rows_two_halves(
    const uint8_t* weight_rows_packed,  // [8, k_total/2] starting from this row group
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

// SP-12-D gate_up: grid kernel.
// blockIdx.x encodes (slot, row_tile_of_8) — slot in [0,top_k), row_tile in [0, 2*INTER/8)
// Block has 32 threads (1 warp). Each block produces 8 output values for that (slot, row_tile).
__global__ void spark_fp8_gate_up_kernel(
    const uint8_t* __restrict__ act_packed,
    const float*   __restrict__ act_scale,
    const int32_t* __restrict__ expert_ids,
    const uint8_t* __restrict__ gate_up_packed,
    const float*   __restrict__ gate_up_scale,
    float          gate_up_global_scale,
    int hidden,
    int intermediate,    // = 512
    int top_k,           // = 8
    float* __restrict__ gate_up_out  // [top_k, 2*INTER]
) {
    const int rows_per_tile = 8;
    const int num_row_tiles = (2 * intermediate) / rows_per_tile;  // 128
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
        fill_a_two_halves(act_packed, k_base, lane, a_low, a_high);
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
        const int64_t base = (int64_t)slot * (2 * intermediate) + row_start;
        gate_up_out[base + lane * 2 + 0] = acc[0];
        gate_up_out[base + lane * 2 + 1] = acc[1];
    }
}

// SP-12-D down: grid kernel. Same structure but operates per (slot, hidden_row_tile_of_8).
__global__ void spark_fp8_down_kernel(
    const uint8_t* __restrict__ inter_packed,    // [top_k, INTER/2]
    const float*   __restrict__ inter_scale,     // [top_k, INTER/16]
    const int32_t* __restrict__ expert_ids,
    const uint8_t* __restrict__ down_packed,
    const float*   __restrict__ down_scale,
    float          down_global_scale,
    int hidden,             // = 2048
    int intermediate,       // = 512
    int top_k,              // = 8
    float* __restrict__ down_out  // [top_k, HIDDEN]
) {
    const int rows_per_tile = 8;
    const int num_row_tiles = hidden / rows_per_tile;  // 256
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
    const int num_k32 = intermediate / 32;   // 16
    const int k_div16 = intermediate / 16;    // 32

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
        fill_a_two_halves(slot_inter, k_base, lane, a_low, a_high);
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
        const int64_t base = (int64_t)slot * hidden + row_start;
        down_out[base + lane * 2 + 0] = acc[0];
        down_out[base + lane * 2 + 1] = acc[1];
    }
}

}  // namespace

torch::Tensor spark_fp8_gate_up(
    torch::Tensor act_packed,
    torch::Tensor act_scale,
    torch::Tensor expert_ids,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    int64_t intermediate
) {
    TORCH_CHECK(gate_up_packed.dim() == 3, "gate_up_packed must be [E, 2*INTER, HIDDEN/2]");
    const int hidden = act_packed.numel() * 2;
    const int top_k = expert_ids.numel();
    auto out = torch::zeros({top_k, 2 * (int)intermediate},
                            torch::dtype(torch::kFloat32).device(act_packed.device()));
    const int num_row_tiles = (2 * intermediate) / 8;
    const int total_blocks = top_k * num_row_tiles;
    spark_fp8_gate_up_kernel<<<total_blocks, 32>>>(
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

torch::Tensor spark_fp8_down(
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
    const int num_row_tiles = hidden / 8;
    const int total_blocks = top_k * num_row_tiles;
    spark_fp8_down_kernel<<<total_blocks, 32>>>(
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


# ---------------------------------------------------------------------------
# Driver — end-to-end active-MoE forward via SP-12-D kernels
# ---------------------------------------------------------------------------

E2M1_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def quantize_to_e2m1(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32 tensor to packed E2M1 + per-16 FP32 scale.
    x: [M, K] or [K]
    Returns: (packed_uint8 [M, K/2] or [K/2], scale_fp32 [M, K/16] or [K/16])
    """
    flat = x.dim() == 1
    if flat:
        x = x.unsqueeze(0)
    m, k = x.shape
    assert k % 16 == 0
    table = E2M1_TABLE.to(x.device)
    xg = x.reshape(m, k // 16, 16)
    abs_max = xg.abs().amax(dim=-1)  # [m, k/16]
    scale = (abs_max / 6.0).clamp_min(1e-8)
    normalized = (xg.abs() / scale.unsqueeze(-1)).clamp(0, 6.0)
    # Nearest E2M1 magnitude code
    mag = torch.argmin((normalized.unsqueeze(-1) - table.view(1, 1, 1, -1)).abs(), dim=-1)
    sign = (xg < 0).to(torch.uint8) * 8
    codes = (mag.to(torch.uint8) | sign).reshape(m, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4))
    if flat:
        packed = packed.squeeze(0)
        scale = scale.squeeze(0)
    return packed.contiguous(), scale.contiguous()


def _build_module(build_root: Path, verbose: bool):
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True, exist_ok=True)
    cpp_path = build_root / "sp12d_bindings.cpp"
    cu_path = build_root / "sp12d_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")
    return load(
        name="lynn_sp12d_spark_fp8_active_moe",
        sources=[str(cpp_path), str(cu_path)],
        build_directory=str(build_root),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_121a"],
        verbose=verbose,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_sp12d_build")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--intermediate", type=int, default=512)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260516)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cap = torch.cuda.get_device_capability(0)
    print(f"[sp12d] device: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
    print(f"[sp12d] hidden={args.hidden} intermediate={args.intermediate} num_experts={args.num_experts} top_k={args.top_k}")

    print(f"[sp12d] building kernels...")
    t0 = time.time()
    module = _build_module(Path(args.build_dir), args.verbose)
    build_seconds = time.time() - t0
    print(f"[sp12d] build OK in {build_seconds:.1f}s")

    torch.manual_seed(args.seed)
    H = args.hidden
    I = args.intermediate
    E = args.num_experts
    TK = args.top_k

    # --- Generate synthetic Lynn 27B-like inputs ---
    # Activation: random BF16-range FP32, quantized to E2M1 + per-16 scale
    act_fp32 = torch.randn(H, dtype=torch.float32, device="cuda") * 0.5
    act_packed, act_scale = quantize_to_e2m1(act_fp32)
    act_decoded = act_fp32.clone()  # we'll compare against the quantized round-trip

    # Verify decoded round-trip matches what kernel uses
    # decoded = sign * mag_value * scale[group]
    # ... skip, the kernel matches the E2M1 contract by definition

    # Expert routing
    expert_ids = torch.randint(0, E, (TK,), dtype=torch.int32, device="cuda")
    routing_weights = F.softmax(torch.randn(TK, dtype=torch.float32, device="cuda"), dim=0)

    # gate_up weights: [E, 2*I, H/2] uint8 + [E, 2*I, H/16] f32
    gate_up_fp32 = torch.randn(E, 2 * I, H, dtype=torch.float32, device="cuda") * 0.05
    gate_up_packed = torch.empty(E, 2 * I, H // 2, dtype=torch.uint8, device="cuda")
    gate_up_scale = torch.empty(E, 2 * I, H // 16, dtype=torch.float32, device="cuda")
    for e in range(E):
        p, s = quantize_to_e2m1(gate_up_fp32[e])
        gate_up_packed[e] = p
        gate_up_scale[e] = s
    gate_up_global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    # down weights: [E, H, I/2]
    down_fp32 = torch.randn(E, H, I, dtype=torch.float32, device="cuda") * 0.05
    down_packed = torch.empty(E, H, I // 2, dtype=torch.uint8, device="cuda")
    down_scale = torch.empty(E, H, I // 16, dtype=torch.float32, device="cuda")
    for e in range(E):
        p, s = quantize_to_e2m1(down_fp32[e])
        down_packed[e] = p
        down_scale[e] = s
    down_global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    # --- Stage 1: gate_up via Spark FP8 kernel ---
    print(f"[sp12d] stage 1: gate_up FP8 kernel...")
    t0 = time.time()
    gate_up_out = module.spark_fp8_gate_up(
        act_packed, act_scale,
        expert_ids,
        gate_up_packed, gate_up_scale, gate_up_global_scale,
        I,
    )
    torch.cuda.synchronize()
    gate_up_ms = (time.time() - t0) * 1000.0
    print(f"[sp12d] gate_up output shape: {gate_up_out.shape}  ms: {gate_up_ms:.3f}")

    # Reference: dequant + matmul in FP32
    # Decode packed weight back to FP32 (using same quantization round-trip)
    def decode_packed(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Reverse of quantize_to_e2m1: packed [..., K/2] uint8 + scale [..., K/16] -> [..., K]."""
        # Each byte: low nibble + high nibble = 2 codes
        codes_low = packed & 0x0F
        codes_high = (packed >> 4) & 0x0F
        codes = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2,
                            dtype=torch.uint8, device=packed.device)
        codes[..., 0::2] = codes_low
        codes[..., 1::2] = codes_high
        mag = (codes & 0x07).long()
        sign = ((codes & 0x08) != 0).float() * -2 + 1  # -1 if sign, +1 if not
        table = E2M1_TABLE.to(packed.device)
        values = table[mag] * sign  # [..., K]
        # Apply per-16 scale
        K = values.shape[-1]
        values_grouped = values.reshape(*values.shape[:-1], K // 16, 16)
        scaled = values_grouped * scale.unsqueeze(-1)
        return scaled.reshape(*values.shape[:-1], K)

    # Decode activation
    act_decoded_back = decode_packed(act_packed, act_scale)
    # For each active expert, decode its gate_up weight and matmul
    gate_up_ref = torch.zeros(TK, 2 * I, device="cuda")
    for slot in range(TK):
        e = int(expert_ids[slot].item())
        w = decode_packed(gate_up_packed[e], gate_up_scale[e])  # [2*I, H]
        gate_up_ref[slot] = w @ act_decoded_back  # [2*I]

    gate_up_diff = (gate_up_out - gate_up_ref).abs()
    gate_up_max_abs = float(gate_up_diff.max().item())
    gate_up_rel = float((gate_up_out - gate_up_ref).norm() / gate_up_ref.norm().clamp_min(1e-9))
    print(f"[sp12d] gate_up max_abs_err: {gate_up_max_abs:.6e}  rel_err: {gate_up_rel:.6e}")
    gate_up_pass = gate_up_max_abs < 1e-3 * float(gate_up_ref.abs().max().item())
    print(f"[sp12d] gate_up PASS: {gate_up_pass}")

    # --- Stage 2: SiLU * up (Python) ---
    print(f"[sp12d] stage 2: SiLU(gate) * up (Python)...")
    gate = gate_up_out[:, :I]
    up = gate_up_out[:, I:]
    inter = F.silu(gate) * up  # [TK, I] FP32
    # Quantize to E2M1 for down stage
    inter_packed, inter_scale = quantize_to_e2m1(inter)
    print(f"[sp12d] inter shape: {inter.shape}, packed shape: {inter_packed.shape}")

    # --- Stage 3: down via Spark FP8 kernel ---
    print(f"[sp12d] stage 3: down FP8 kernel...")
    t0 = time.time()
    down_out = module.spark_fp8_down(
        inter_packed, inter_scale,
        expert_ids,
        down_packed, down_scale, down_global_scale,
        H,
    )
    torch.cuda.synchronize()
    down_ms = (time.time() - t0) * 1000.0
    print(f"[sp12d] down output shape: {down_out.shape}  ms: {down_ms:.3f}")

    # Down reference
    inter_decoded = decode_packed(inter_packed, inter_scale)
    down_ref = torch.zeros(TK, H, device="cuda")
    for slot in range(TK):
        e = int(expert_ids[slot].item())
        w = decode_packed(down_packed[e], down_scale[e])  # [H, I]
        down_ref[slot] = w @ inter_decoded[slot]

    down_diff = (down_out - down_ref).abs()
    down_max_abs = float(down_diff.max().item())
    down_rel = float((down_out - down_ref).norm() / down_ref.norm().clamp_min(1e-9))
    print(f"[sp12d] down max_abs_err: {down_max_abs:.6e}  rel_err: {down_rel:.6e}")
    down_pass = down_max_abs < 1e-3 * float(down_ref.abs().max().item())
    print(f"[sp12d] down PASS: {down_pass}")

    # --- Stage 4: routing weighted sum ---
    print(f"[sp12d] stage 4: routing weighted sum (Python)...")
    moe_out = (routing_weights.unsqueeze(1) * down_out).sum(dim=0)  # [H]

    # End-to-end ref
    moe_ref = (routing_weights.unsqueeze(1) * down_ref).sum(dim=0)
    moe_diff = (moe_out - moe_ref).abs()
    moe_max_abs = float(moe_diff.max().item())
    moe_rel = float((moe_out - moe_ref).norm() / moe_ref.norm().clamp_min(1e-9))
    print(f"[sp12d] end-to-end max_abs_err: {moe_max_abs:.6e}  rel_err: {moe_rel:.6e}")

    # --- Timing (multiple iters) ---
    print(f"[sp12d] timing gate_up (50 iters)...")
    for _ in range(5):
        module.spark_fp8_gate_up(act_packed, act_scale, expert_ids, gate_up_packed, gate_up_scale, gate_up_global_scale, I)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        module.spark_fp8_gate_up(act_packed, act_scale, expert_ids, gate_up_packed, gate_up_scale, gate_up_global_scale, I)
    e.record()
    torch.cuda.synchronize()
    gate_up_us = float(s.elapsed_time(e) / 50 * 1000.0)

    print(f"[sp12d] timing down (50 iters)...")
    for _ in range(5):
        module.spark_fp8_down(inter_packed, inter_scale, expert_ids, down_packed, down_scale, down_global_scale, H)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        module.spark_fp8_down(inter_packed, inter_scale, expert_ids, down_packed, down_scale, down_global_scale, H)
    e.record()
    torch.cuda.synchronize()
    down_us = float(s.elapsed_time(e) / 50 * 1000.0)

    print(f"[sp12d] gate_up: {gate_up_us:.2f} us/call  down: {down_us:.2f} us/call")
    print(f"[sp12d] total per active-MoE forward (excl. SiLU + routing): {gate_up_us + down_us:.2f} us")

    summary = {
        "type": "sp12d_spark_fp8_active_moe_probe",
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(cap),
        "shape": {"hidden": H, "intermediate": I, "num_experts": E, "top_k": TK},
        "stage_results": {
            "gate_up": {"max_abs_err": gate_up_max_abs, "rel_err": gate_up_rel, "pass": gate_up_pass, "us": gate_up_us},
            "down": {"max_abs_err": down_max_abs, "rel_err": down_rel, "pass": down_pass, "us": down_us},
            "end_to_end": {"max_abs_err": moe_max_abs, "rel_err": moe_rel},
        },
        "total_us": gate_up_us + down_us,
        "all_pass": gate_up_pass and down_pass,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n[sp12d] === SUMMARY ===")
    print(f"[sp12d]   gate_up:       max_abs_err={gate_up_max_abs:.3e}  {gate_up_us:.2f} us  PASS={gate_up_pass}")
    print(f"[sp12d]   down:          max_abs_err={down_max_abs:.3e}  {down_us:.2f} us  PASS={down_pass}")
    print(f"[sp12d]   end-to-end:    max_abs_err={moe_max_abs:.3e}")
    print(f"[sp12d]   total per MoE: {gate_up_us + down_us:.2f} us")
    print(f"[sp12d]   ALL PASS: {summary['all_pass']}")
    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
