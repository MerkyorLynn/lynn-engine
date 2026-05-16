#!/usr/bin/env python3
"""SP-12-C: Spark sm_121 FP8 MMA — 8 output rows × K=2048 in one MMA tile.

Production shape: matches Codex P90 (8 gate rows + 8 up rows × K=2048,
max_abs_err 2.38e-07 on R6000 FP4 block_scale). SP-12-C replicates this
shape using sm_121 FP8 MMA path.

Key change from SP-12-B (1 broadcast row):
  - 8 distinct weight rows loaded into B operand
  - Each lane t loads B[n = t/4, k = (t%4)*4 + ...] — 4 lanes share each n-row
  - After MMA, lanes 0..3 hold all 8 m=0 output values:
      lane 0: out[n=0,1]   lane 1: out[n=2,3]
      lane 2: out[n=4,5]   lane 3: out[n=6,7]
  - Lanes 4..31 compute redundant m=1..7 outputs (single-token waste; will
    be reclaimed in SP-12-D production kernel via token-batching)

PASS gate: max_abs_err < 1e-5 vs scalar reference for all 8 output rows.
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
from torch.utils.cpp_extension import load


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor sp12c_fp8_8row_probe(
    torch::Tensor act_e2m1_packed,    // [K/2] uint8
    torch::Tensor act_scale,          // [K/16] f32
    torch::Tensor weight_e2m1_packed, // [8, K/2] uint8 — 8 output rows
    torch::Tensor weight_scale,       // [8, K/16] f32
    torch::Tensor weight_global_scale // [1] f32
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_8row_probe", &sp12c_fp8_8row_probe,
        "SP-12-C FP8 MMA 8-row tile, K=2048");
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
    float d[4], const float c[4]
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
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

// A operand for broadcast activation across m=16, split-16 zero-pad halves.
__device__ __forceinline__ void fill_a_two_halves(
    const uint8_t* act_packed,
    int k_base, int lane,
    uint32_t a_low[4], uint32_t a_high[4]
) {
    const int k0 = k_base + (lane & 3) * 4;
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

    // Broadcast across m=0..7 (regs 0,2) and m=8..15 (regs 1,3)
    a_low[0] = r_low;
    a_low[1] = r_low;
    a_low[2] = 0;
    a_low[3] = 0;

    a_high[0] = 0;
    a_high[1] = 0;
    a_high[2] = r_high;
    a_high[3] = r_high;
}

// B operand for 8 distinct output rows. Each lane t loads from row = t/4.
__device__ __forceinline__ void fill_b_8rows_two_halves(
    const uint8_t* weight_packed,  // [8, K/2] flat: 8 rows × K/2 bytes
    int k_total,                   // total K dim = 2048
    int k_base,
    int lane,
    uint32_t b_low[2],
    uint32_t b_high[2]
) {
    const int n_row = lane >> 2;            // 0..7
    const uint8_t* row_ptr = weight_packed + n_row * (k_total / 2);
    const int k0 = k_base + (lane & 3) * 4;
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

// Look up per-(n_col, k_half) weight scale.
//
// Critical: m16n8 D layout for thread t is:
//   d[0]: D[m=t/4,     n=(t%4)*2 + 0]   <- n_a column
//   d[1]: D[m=t/4,     n=(t%4)*2 + 1]   <- n_b column
//   d[2]: D[m=t/4 + 8, n=(t%4)*2 + 0]   <- same n as d[0]
//   d[3]: D[m=t/4 + 8, n=(t%4)*2 + 1]   <- same n as d[1]
//
// Each lane's d[0]/d[2] use weight row (t%4)*2's scale, and d[1]/d[3] use
// weight row (t%4)*2+1's scale. SP-12-C v1 incorrectly used lane/4 (the
// B-load row) for all four d[]; lanes 0..3 all load row 0 for B, but the
// MMA hardware shuffles so their d[1] is actually the dot product against
// B[row 1] data loaded by lanes 4..7.
__device__ __forceinline__ float weight_scale_for_n(
    const float* weight_scale,  // [8, K/16] flat
    int k_total,
    int k_base,
    int n_col,    // 0..7 (which output row's scale)
    int half      // 0 for low half (k_base..k_base+15), 1 for high (k_base+16..k_base+31)
) {
    const int scale_idx = k_base / 16 + half;
    return weight_scale[n_col * (k_total / 16) + scale_idx];
}

// SP-12-C kernel: single warp, 8 output rows, K=2048 split-16.
//
// Output layout: 8 floats out[0..7] corresponding to D[m=0, n=0..7] after MMA.
// After MMA, only lanes 0..3 hold the m=0 outputs we care about:
//   lane 0: d[0]=D[0,0], d[1]=D[0,1]
//   lane 1: d[0]=D[0,2], d[1]=D[0,3]
//   lane 2: d[0]=D[0,4], d[1]=D[0,5]
//   lane 3: d[0]=D[0,6], d[1]=D[0,7]
//
// Lanes 4..31 also compute m=1..7 (redundant under broadcast A).

__global__ void sp12c_kernel(
    const uint8_t* __restrict__ act_packed,
    const float* __restrict__ act_scale,
    const uint8_t* __restrict__ weight_packed,    // [8, K/2]
    const float* __restrict__ weight_scale,       // [8, K/16]
    float weight_global_scale,
    int k_total,
    float* __restrict__ out  // [8] f32
) {
    const int lane = threadIdx.x;
    // d[0,2] -> n_col_a = (lane%4)*2 + 0
    // d[1,3] -> n_col_b = (lane%4)*2 + 1
    const int n_col_a = (lane & 3) * 2 + 0;
    const int n_col_b = (lane & 3) * 2 + 1;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const int num_k32 = k_total / 32;

    #pragma unroll 1
    for (int t = 0; t < num_k32; ++t) {
        const int k_base = t * 32;
        const float a_scale_low  = act_scale[k_base / 16];
        const float a_scale_high = act_scale[k_base / 16 + 1];
        // Per-n-col weight scales (d[0]/d[2] vs d[1]/d[3] use different weight rows)
        const float w_scale_a_low  = weight_scale_for_n(weight_scale, k_total, k_base, n_col_a, 0);
        const float w_scale_b_low  = weight_scale_for_n(weight_scale, k_total, k_base, n_col_b, 0);
        const float w_scale_a_high = weight_scale_for_n(weight_scale, k_total, k_base, n_col_a, 1);
        const float w_scale_b_high = weight_scale_for_n(weight_scale, k_total, k_base, n_col_b, 1);

        uint32_t a_low[4], a_high[4];
        uint32_t b_low[2], b_high[2];
        fill_a_two_halves(act_packed, k_base, lane, a_low, a_high);
        fill_b_8rows_two_halves(weight_packed, k_total, k_base, lane, b_low, b_high);

        float d_low[4], d_high[4];
        float zero4[4] = {0.f, 0.f, 0.f, 0.f};
        fp8_mma_m16n8k32(a_low,  b_low,  d_low,  zero4);
        fp8_mma_m16n8k32(a_high, b_high, d_high, zero4);

        // Apply per-(n_col, k_half) scales
        const float scale_a_low  = a_scale_low  * w_scale_a_low  / weight_global_scale;
        const float scale_b_low  = a_scale_low  * w_scale_b_low  / weight_global_scale;
        const float scale_a_high = a_scale_high * w_scale_a_high / weight_global_scale;
        const float scale_b_high = a_scale_high * w_scale_b_high / weight_global_scale;

        acc[0] += d_low[0] * scale_a_low + d_high[0] * scale_a_high;  // n_col_a
        acc[1] += d_low[1] * scale_b_low + d_high[1] * scale_b_high;  // n_col_b
        acc[2] += d_low[2] * scale_a_low + d_high[2] * scale_a_high;  // same n as d[0]
        acc[3] += d_low[3] * scale_b_low + d_high[3] * scale_b_high;  // same n as d[1]
    }

    // Lanes 0..3 write the 8 output values for m=0.
    // Lane 0 -> out[0], out[1]   (n=0, n=1)
    // Lane 1 -> out[2], out[3]
    // Lane 2 -> out[4], out[5]
    // Lane 3 -> out[6], out[7]
    if (lane < 4) {
        out[lane * 2 + 0] = acc[0];
        out[lane * 2 + 1] = acc[1];
    }
}

}  // namespace

torch::Tensor sp12c_fp8_8row_probe(
    torch::Tensor act_e2m1_packed,
    torch::Tensor act_scale,
    torch::Tensor weight_e2m1_packed,
    torch::Tensor weight_scale,
    torch::Tensor weight_global_scale
) {
    TORCH_CHECK(act_e2m1_packed.dtype() == torch::kUInt8 && act_e2m1_packed.is_cuda(),
                "act must be cuda uint8");
    TORCH_CHECK(weight_e2m1_packed.dtype() == torch::kUInt8, "weight must be uint8");
    TORCH_CHECK(weight_e2m1_packed.is_cuda(), "weight must be cuda");
    TORCH_CHECK(weight_e2m1_packed.dim() == 2 && weight_e2m1_packed.size(0) == 8,
                "weight must be [8, K/2]");

    const int64_t k_total = act_e2m1_packed.numel() * 2;
    TORCH_CHECK(k_total % 32 == 0, "K must be multiple of 32");
    TORCH_CHECK(weight_e2m1_packed.size(1) == k_total / 2, "weight K mismatch");
    TORCH_CHECK(act_scale.numel() == k_total / 16, "act_scale shape");
    TORCH_CHECK(weight_scale.dim() == 2 && weight_scale.size(0) == 8 && weight_scale.size(1) == k_total / 16,
                "weight_scale must be [8, K/16]");

    auto out = torch::zeros({8}, torch::dtype(torch::kFloat32).device(act_e2m1_packed.device()));

    sp12c_kernel<<<1, 32>>>(
        act_e2m1_packed.data_ptr<uint8_t>(),
        act_scale.data_ptr<float>(),
        weight_e2m1_packed.contiguous().data_ptr<uint8_t>(),
        weight_scale.contiguous().data_ptr<float>(),
        weight_global_scale.item<float>(),
        (int)k_total,
        out.data_ptr<float>()
    );

    return out;
}
"""


E2M1_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1_decode(code: int) -> float:
    mag = code & 0x07
    sign = (code & 0x08) != 0
    v = E2M1_MAGNITUDES[mag]
    return -v if sign else v


def e2m1_decode_packed_to_tensor(packed: torch.Tensor) -> torch.Tensor:
    out = []
    for byte in packed.cpu().numpy():
        out.append(e2m1_decode(byte & 0x0F))
        out.append(e2m1_decode((byte >> 4) & 0x0F))
    return torch.tensor(out, dtype=torch.float32)


def scalar_reference_8rows(
    act_packed: torch.Tensor,
    act_scale: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: float,
) -> torch.Tensor:
    """Scalar reference for 8 output rows."""
    k_total = act_packed.numel() * 2
    a_decoded = e2m1_decode_packed_to_tensor(act_packed)   # [K]
    a_scaled = a_decoded.clone()
    for g in range(k_total // 16):
        a_scaled[g*16:(g+1)*16] *= act_scale[g].item()

    out = torch.zeros(8, dtype=torch.float64)  # use double for reference precision
    for n in range(8):
        w_decoded = e2m1_decode_packed_to_tensor(weight_packed[n])
        w_scaled = w_decoded.clone()
        for g in range(k_total // 16):
            w_scaled[g*16:(g+1)*16] *= weight_scale[n, g].item() / weight_global_scale
        out[n] = (a_scaled.double() * w_scaled.double()).sum()
    return out.float()


def _build_module(build_root: Path, verbose: bool):
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True, exist_ok=True)
    cpp_path = build_root / "sp12c_bindings.cpp"
    cu_path = build_root / "sp12c_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")
    return load(
        name="lynn_sp12c_fp8_8row_probe",
        sources=[str(cpp_path), str(cu_path)],
        build_directory=str(build_root),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_121a"],
        verbose=verbose,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_sp12c_build")
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20260516)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cap = torch.cuda.get_device_capability(0)
    print(f"[sp12c] device: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
    print(f"[sp12c] torch: {torch.__version__}  cuda: {torch.version.cuda}")

    K = args.k
    print(f"[sp12c] K = {K} (8 output rows)")

    print(f"[sp12c] building...")
    t0 = time.time()
    module = _build_module(Path(args.build_dir), args.verbose)
    print(f"[sp12c] build OK in {time.time() - t0:.1f}s")

    torch.manual_seed(args.seed)

    # Generate synthetic but realistic Lynn-like inputs
    act_nibbles = torch.randint(0, 16, (K,), dtype=torch.uint8)
    act_bytes = (act_nibbles[0::2] | (act_nibbles[1::2] << 4)).contiguous()
    act_packed = act_bytes.cuda()
    act_scale = (0.1 + 0.4 * torch.rand(K // 16)).cuda().contiguous()

    weight_nibbles = torch.randint(0, 16, (8, K), dtype=torch.uint8)
    weight_bytes = torch.empty((8, K // 2), dtype=torch.uint8)
    for n in range(8):
        weight_bytes[n] = (weight_nibbles[n, 0::2] | (weight_nibbles[n, 1::2] << 4))
    weight_packed = weight_bytes.cuda()
    weight_scale = (0.001 + 0.009 * torch.rand(8, K // 16)).cuda().contiguous()

    weight_global_scale = torch.tensor([1.0], dtype=torch.float32).cuda()

    print(f"[sp12c] running CUDA kernel...")
    out_cuda = module.fp8_8row_probe(
        act_packed, act_scale,
        weight_packed, weight_scale,
        weight_global_scale,
    )
    out_cuda_cpu = out_cuda.cpu()
    print(f"[sp12c] CUDA output: {out_cuda_cpu.tolist()}")

    print(f"[sp12c] computing scalar reference (8 rows × K=2048)...")
    out_ref = scalar_reference_8rows(
        act_packed.cpu(), act_scale.cpu(),
        weight_packed.cpu(), weight_scale.cpu(),
        1.0,
    )
    print(f"[sp12c] reference:   {out_ref.tolist()}")

    diff = (out_cuda_cpu - out_ref).abs()
    max_abs_err = float(diff.max().item())
    rel_err_per_row = (diff / out_ref.abs().clamp_min(1e-9))
    max_rel_err = float(rel_err_per_row.max().item())
    print(f"[sp12c] max_abs_err = {max_abs_err:.6e}")
    print(f"[sp12c] max_rel_err = {max_rel_err:.6e}")

    # Timing
    print(f"[sp12c] timing 500 iterations...")
    for _ in range(10):
        module.fp8_8row_probe(act_packed, act_scale, weight_packed, weight_scale, weight_global_scale)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    iters = 500
    for _ in range(iters):
        module.fp8_8row_probe(act_packed, act_scale, weight_packed, weight_scale, weight_global_scale)
    end.record()
    torch.cuda.synchronize()
    per_call_us = float(start.elapsed_time(end) / iters * 1000.0)
    print(f"[sp12c] timing: {per_call_us:.3f} us/call (8 rows × K={K}, single warp)")
    print(f"[sp12c] per-row: {per_call_us / 8:.3f} us")

    pass_gate = max_abs_err < 1e-5
    summary = {
        "type": "sp12c_sm121_fp8_8row_tile_probe",
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(cap),
        "K": K,
        "n_rows": 8,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "cuda_output": out_cuda_cpu.tolist(),
        "scalar_reference": out_ref.tolist(),
        "per_call_us": per_call_us,
        "per_row_us": per_call_us / 8,
        "pass": pass_gate,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n[sp12c] === SUMMARY ===")
    print(f"[sp12c]   max_abs_err = {max_abs_err:.6e}")
    print(f"[sp12c]   max_rel_err = {max_rel_err:.6e}")
    print(f"[sp12c]   per-call    = {per_call_us:.3f} us  (per-row: {per_call_us/8:.3f} us)")
    print(f"[sp12c]   PASS = {pass_gate}")
    return 0 if pass_gate else 2


if __name__ == "__main__":
    sys.exit(main())
