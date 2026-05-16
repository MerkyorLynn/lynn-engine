#!/usr/bin/env python3
"""SP-12-A: Spark sm_121 FP8 MMA + E2M1->FP8 LUT tile contract probe.

Minimal viable kernel that proves the Spark-specific FP8 path:

  Lynn-native E2M1 weight bytes
      | LUT-map to FP8 E4M3 (lossless, all 16 E2M1 values fit in FP8 range)
  Lynn-native E2M1 activation bytes
      | LUT-map to FP8 E4M3
      |
      v
  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32  (Spark sm_121 PASS, SP-11)
      |
      v
  FP32 accumulator
      |
      v
  Compare against scalar reference using the SAME E2M1 LUT decoded to float

SP-12-A intentionally:
  1. Uses SYNTHETIC inputs (all-1 magnitudes, varied signs) for first parity gate
  2. Does NOT apply per-16 scales (that's SP-12-B)
  3. Does NOT load real Lynn 27B weights yet (that's SP-12-B)
  4. Single block, single warp, one m16n8k32 tile = 16x8 FP32 output

Goal: prove the FP8 MMA path bit-exact-matches the scalar reference on a tile
of E2M1 codes mapped to FP8 E4M3 via lossless LUT.

If SP-12-A passes max_abs_err <= 1e-5, we have the FP8 MMA primitive ready and
SP-12-B can add real Lynn weights + per-16 FP32 scale epilogue.
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


# ---------------------------------------------------------------------------
# E2M1 -> FP8 E4M3 LUT (lossless: all 16 E2M1 values are exact FP8 E4M3 values)
# ---------------------------------------------------------------------------
#
# E2M1 nibble layout (Lynn-native):
#   bit 3: sign (1 = negative)
#   bits 2..0: magnitude code (0..7)
#
# E2M1 magnitudes: {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
#
# E4M3 (FP8): sign + 4 exp bits + 3 mantissa bits, range -448 to 448
#
# Mapping (each magnitude -> exact E4M3 byte):
#
#   +0.0 -> 0x00    +0.5 -> 0x30    +1.0 -> 0x38    +1.5 -> 0x3C
#   +2.0 -> 0x40    +3.0 -> 0x44    +4.0 -> 0x48    +6.0 -> 0x4C
#   -0.0 -> 0x80    -0.5 -> 0xB0    -1.0 -> 0xB8    -1.5 -> 0xBC
#   -2.0 -> 0xC0    -3.0 -> 0xC4    -4.0 -> 0xC8    -6.0 -> 0xCC

E2M1_TO_E4M3_LUT = [
    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,  # +0, +0.5, +1, +1.5, +2, +3, +4, +6
    0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC,  # -0, -0.5, -1, -1.5, -2, -3, -4, -6
]
E2M1_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1_code_to_float(code: int) -> float:
    """Decode E2M1 nibble to float (reference)."""
    mag = code & 0x07
    sign = (code & 0x08) != 0
    v = E2M1_MAGNITUDES[mag]
    return -v if sign else v


# ---------------------------------------------------------------------------
# CUDA source: FP8 MMA + E2M1->FP8 LUT tile kernel
# ---------------------------------------------------------------------------

CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor sp12a_fp8_tile_probe(
    torch::Tensor act_e2m1_packed,    // [16] uint8 (32 nibbles = 32 K elements)
    torch::Tensor weight_e2m1_packed  // [16 * 16] uint8 (16 rows x 32 K nibbles = 16*32/2 = 256 bytes)
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_tile_probe", &sp12a_fp8_tile_probe,
        "SP-12-A FP8 E4M3 MMA tile probe with E2M1->FP8 LUT");
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

// Lossless LUT: E2M1 nibble (0..15) -> FP8 E4M3 byte
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

// Pack 4 FP8 E4M3 bytes into one uint32 (little-endian: byte 0 in LSB)
__device__ __forceinline__ uint32_t pack4_fp8(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    return (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
}

// Build A operand for one lane of m16n8k32 e4m3 MMA.
// Standard CUTLASS / NVIDIA layout for m16n8k32:
//   A is [16 m, 32 k] row-major.
//   Thread t (0..31) holds 16 FP8 elements (4 uint32 regs):
//     regs[0] = A[m=t/4 +  0, k=(t%4)*4..(t%4)*4+3]   (k offset 0)
//     regs[1] = A[m=t/4 +  8, k=(t%4)*4..(t%4)*4+3]   (k offset 0)
//     regs[2] = A[m=t/4 +  0, k=(t%4)*4+16..(t%4)*4+19] (k offset 16)
//     regs[3] = A[m=t/4 +  8, k=(t%4)*4+16..(t%4)*4+19] (k offset 16)
__device__ __forceinline__ void fill_a_words(
    const uint8_t* act_packed,     // [K/2] E2M1 packed; here K=32 so 16 bytes
    int lane,
    uint32_t out[4]
) {
    // For sp12a, A is fed from a "fake" m=16 broadcast of the activation.
    // We replicate the single activation vector across all 16 m rows for the
    // probe — this is a contrived test, just to verify FP8 MMA + LUT works.
    //
    // So A[m=any, k] = act[k].
    //
    // Each lane needs 4 regs:
    //   reg0: k = (t%4)*4 + 0..3
    //   reg1: k = (t%4)*4 + 0..3 (different m, same k -> same data after broadcast)
    //   reg2: k = (t%4)*4 + 16..19
    //   reg3: k = (t%4)*4 + 16..19
    //
    // Wait — in the broadcast case, reg0 == reg1 and reg2 == reg3. Let me just
    // compute reg0 and reg2 properly and replicate.

    const int k_base_low  = (lane & 3) * 4;
    const int k_base_high = (lane & 3) * 4 + 16;

    uint8_t b0_low  = e2m1_to_e4m3(get_nibble(act_packed, k_base_low + 0));
    uint8_t b1_low  = e2m1_to_e4m3(get_nibble(act_packed, k_base_low + 1));
    uint8_t b2_low  = e2m1_to_e4m3(get_nibble(act_packed, k_base_low + 2));
    uint8_t b3_low  = e2m1_to_e4m3(get_nibble(act_packed, k_base_low + 3));

    uint8_t b0_high = e2m1_to_e4m3(get_nibble(act_packed, k_base_high + 0));
    uint8_t b1_high = e2m1_to_e4m3(get_nibble(act_packed, k_base_high + 1));
    uint8_t b2_high = e2m1_to_e4m3(get_nibble(act_packed, k_base_high + 2));
    uint8_t b3_high = e2m1_to_e4m3(get_nibble(act_packed, k_base_high + 3));

    uint32_t reg_low  = pack4_fp8(b0_low,  b1_low,  b2_low,  b3_low);
    uint32_t reg_high = pack4_fp8(b0_high, b1_high, b2_high, b3_high);

    out[0] = reg_low;
    out[1] = reg_low;   // broadcast across m
    out[2] = reg_high;
    out[3] = reg_high;
}

// Build B operand for one lane.
// B is [8 n, 32 k] col-major (so logically B = W^T where W is [32 k, 8 n]).
// For our use: B is [N=8 rows, K=32], representing the first 8 rows of W in row-major.
// Thread t holds 8 FP8 elements (2 uint32):
//   reg0: B[n=t/4, k=(t%4)*4..(t%4)*4+3]
//   reg1: B[n=t/4, k=(t%4)*4+16..(t%4)*4+19]
__device__ __forceinline__ void fill_b_words(
    const uint8_t* weight_packed,  // [N*K/2] E2M1 packed; here N=8, K=32 -> 128 bytes
    int lane,
    uint32_t out[2]
) {
    const int n_row = lane >> 2;  // 0..7
    const int k_base_low  = (lane & 3) * 4;
    const int k_base_high = (lane & 3) * 4 + 16;
    const uint8_t* row_ptr = weight_packed + n_row * (32 / 2);  // 16 bytes per row

    uint8_t b0_low = e2m1_to_e4m3(get_nibble(row_ptr, k_base_low + 0));
    uint8_t b1_low = e2m1_to_e4m3(get_nibble(row_ptr, k_base_low + 1));
    uint8_t b2_low = e2m1_to_e4m3(get_nibble(row_ptr, k_base_low + 2));
    uint8_t b3_low = e2m1_to_e4m3(get_nibble(row_ptr, k_base_low + 3));

    uint8_t b0_high = e2m1_to_e4m3(get_nibble(row_ptr, k_base_high + 0));
    uint8_t b1_high = e2m1_to_e4m3(get_nibble(row_ptr, k_base_high + 1));
    uint8_t b2_high = e2m1_to_e4m3(get_nibble(row_ptr, k_base_high + 2));
    uint8_t b3_high = e2m1_to_e4m3(get_nibble(row_ptr, k_base_high + 3));

    out[0] = pack4_fp8(b0_low,  b1_low,  b2_low,  b3_low);
    out[1] = pack4_fp8(b0_high, b1_high, b2_high, b3_high);
}

// Single-warp m16n8k32 FP8 E4M3 MMA tile kernel.
// Output: [16, 8] FP32 = m x n.
__global__ void sp12a_fp8_tile_kernel(
    const uint8_t* __restrict__ act_packed,    // [K/2] E2M1 packed activation
    const uint8_t* __restrict__ weight_packed, // [N*K/2] E2M1 packed weight (N=8 first rows)
    float* __restrict__ out                    // [16, 8] FP32 output
) {
    const int lane = threadIdx.x;

    uint32_t a_regs[4];
    uint32_t b_regs[2];
    fill_a_words(act_packed, lane, a_regs);
    fill_b_words(weight_packed, lane, b_regs);

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
          "r"(b_regs[0]), "r"(b_regs[1]),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f)
    );

    // Standard m16n8k32 D layout per thread:
    //   d0,d1 -> rows (t/4, t/4+8), col t%4*2 + 0..1   (group 0)
    //   d2,d3 -> same rows, col t%4*2 + 8 .. + 9        (group 1)
    // Wait — that's m16n8 which is m=16 rows, n=8 cols. n=8 / 4 threads-per-col = 2 cols per thread.
    // Standard layout for m16n8 D:
    //   t -> (m=t/4 + 0, n=t%4*2 + 0)  -> d0
    //   t -> (m=t/4 + 0, n=t%4*2 + 1)  -> d1
    //   t -> (m=t/4 + 8, n=t%4*2 + 0)  -> d2
    //   t -> (m=t/4 + 8, n=t%4*2 + 1)  -> d3

    const int m0 = lane >> 2;       // 0..7
    const int m1 = m0 + 8;          // 8..15
    const int n0 = (lane & 3) * 2;
    const int n1 = n0 + 1;

    out[m0 * 8 + n0] = d0;
    out[m0 * 8 + n1] = d1;
    out[m1 * 8 + n0] = d2;
    out[m1 * 8 + n1] = d3;
}

}  // namespace

torch::Tensor sp12a_fp8_tile_probe(
    torch::Tensor act_e2m1_packed,
    torch::Tensor weight_e2m1_packed
) {
    TORCH_CHECK(act_e2m1_packed.is_cuda(), "act must be CUDA");
    TORCH_CHECK(weight_e2m1_packed.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(act_e2m1_packed.dtype() == torch::kUInt8, "act must be uint8");
    TORCH_CHECK(weight_e2m1_packed.dtype() == torch::kUInt8, "weight must be uint8");
    TORCH_CHECK(act_e2m1_packed.numel() == 16, "act must be 16 bytes = 32 E2M1 nibbles");
    TORCH_CHECK(weight_e2m1_packed.numel() == 128, "weight must be 128 bytes = 8 rows x 32 nibbles");

    auto out = torch::zeros({16, 8}, torch::dtype(torch::kFloat32).device(act_e2m1_packed.device()));

    sp12a_fp8_tile_kernel<<<1, 32>>>(
        act_e2m1_packed.data_ptr<uint8_t>(),
        weight_e2m1_packed.data_ptr<uint8_t>(),
        out.data_ptr<float>()
    );

    return out;
}
"""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _build_module(build_root: Path, verbose: bool):
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True, exist_ok=True)
    cpp_path = build_root / "sp12a_bindings.cpp"
    cu_path = build_root / "sp12a_kernel.cu"
    cpp_path.write_text(CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(CUDA_SOURCE, encoding="utf-8")
    return load(
        name="lynn_sp12a_fp8_tile_probe",
        sources=[str(cpp_path), str(cu_path)],
        build_directory=str(build_root),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_121a"],
        verbose=verbose,
    )


def _scalar_reference(act_nibbles: list[int], weight_rows: list[list[int]]) -> torch.Tensor:
    """Scalar reference: same LUT decode to float, then matmul.

    act_nibbles: list of 32 nibble values (0..15)
    weight_rows: list of 8 rows, each 32 nibble values
    Returns: [16, 8] FP32 tensor where row m has act @ weight (broadcast across m).
    """
    act_float = [e2m1_code_to_float(c) for c in act_nibbles]   # [32]
    # Compute for n=0..7
    out = torch.zeros((16, 8), dtype=torch.float32)
    for n in range(8):
        weight_row = [e2m1_code_to_float(c) for c in weight_rows[n]]  # [32]
        dot = sum(a * w for a, w in zip(act_float, weight_row))
        out[:, n] = dot  # broadcast across all 16 m rows
    return out


def _pack_nibbles_to_bytes(nibbles: list[int]) -> bytes:
    """Pack a list of E2M1 nibbles (each 0..15) into bytes (2 nibbles per byte, low-nibble first)."""
    assert len(nibbles) % 2 == 0
    out = bytearray()
    for i in range(0, len(nibbles), 2):
        low = nibbles[i] & 0x0F
        high = nibbles[i + 1] & 0x0F
        out.append(low | (high << 4))
    return bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_sp12a_build")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=20260516)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[sp12a] no CUDA")
        return 1

    cap = torch.cuda.get_device_capability(0)
    print(f"[sp12a] device: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
    print(f"[sp12a] torch: {torch.__version__}  cuda: {torch.version.cuda}")

    print(f"[sp12a] building FP8 MMA + E2M1->FP8 LUT tile kernel (arch=sm_121a)...")
    t_build = time.time()
    module = _build_module(Path(args.build_dir), args.verbose)
    build_seconds = time.time() - t_build
    print(f"[sp12a] build OK in {build_seconds:.1f}s")

    # Generate diverse synthetic inputs
    torch.manual_seed(args.seed)

    # Test case 1: positive constant activation, ramp weight
    print(f"\n[sp12a] === test 1: positive constant act + ramp weight ===")
    act_nibbles = [2] * 32  # all +1.0
    weight_rows = []
    for n in range(8):
        # Each row: ramp through magnitudes
        row = [(k % 8) for k in range(32)]
        weight_rows.append(row)

    act_packed = torch.tensor(list(_pack_nibbles_to_bytes(act_nibbles)), dtype=torch.uint8, device="cuda")
    weight_flat_nibbles = [c for row in weight_rows for c in row]
    weight_packed = torch.tensor(list(_pack_nibbles_to_bytes(weight_flat_nibbles)), dtype=torch.uint8, device="cuda")

    out_cuda = module.fp8_tile_probe(act_packed, weight_packed)
    out_ref = _scalar_reference(act_nibbles, weight_rows).to("cuda")

    diff = (out_cuda - out_ref).abs()
    max_abs_err = float(diff.max().item())
    rel_l2 = float((out_cuda - out_ref).norm() / out_ref.norm().clamp_min(1e-9))

    print(f"[sp12a] test 1 output (CUDA):")
    print(out_cuda[:4, :].cpu().tolist())
    print(f"[sp12a] test 1 output (REF):")
    print(out_ref[:4, :].cpu().tolist())
    print(f"[sp12a] test 1 max_abs_err = {max_abs_err:.6e}")
    print(f"[sp12a] test 1 rel_l2      = {rel_l2:.6e}")

    test1_pass = max_abs_err < 1e-4

    # Test case 2: mixed signs
    print(f"\n[sp12a] === test 2: mixed sign act + mixed sign weight ===")
    act_nibbles_2 = [(k * 3) % 16 for k in range(32)]
    weight_rows_2 = []
    for n in range(8):
        row = [((k + n) * 5) % 16 for k in range(32)]
        weight_rows_2.append(row)

    act_packed_2 = torch.tensor(list(_pack_nibbles_to_bytes(act_nibbles_2)), dtype=torch.uint8, device="cuda")
    weight_flat_2 = [c for row in weight_rows_2 for c in row]
    weight_packed_2 = torch.tensor(list(_pack_nibbles_to_bytes(weight_flat_2)), dtype=torch.uint8, device="cuda")

    out_cuda_2 = module.fp8_tile_probe(act_packed_2, weight_packed_2)
    out_ref_2 = _scalar_reference(act_nibbles_2, weight_rows_2).to("cuda")

    diff_2 = (out_cuda_2 - out_ref_2).abs()
    max_abs_err_2 = float(diff_2.max().item())
    rel_l2_2 = float((out_cuda_2 - out_ref_2).norm() / out_ref_2.norm().clamp_min(1e-9))

    print(f"[sp12a] test 2 max_abs_err = {max_abs_err_2:.6e}")
    print(f"[sp12a] test 2 rel_l2      = {rel_l2_2:.6e}")
    test2_pass = max_abs_err_2 < 1e-4

    # Test case 3: random
    print(f"\n[sp12a] === test 3: random nibbles ===")
    act_nibbles_3 = torch.randint(0, 16, (32,)).tolist()
    weight_rows_3 = [torch.randint(0, 16, (32,)).tolist() for _ in range(8)]
    act_packed_3 = torch.tensor(list(_pack_nibbles_to_bytes(act_nibbles_3)), dtype=torch.uint8, device="cuda")
    weight_flat_3 = [c for row in weight_rows_3 for c in row]
    weight_packed_3 = torch.tensor(list(_pack_nibbles_to_bytes(weight_flat_3)), dtype=torch.uint8, device="cuda")

    out_cuda_3 = module.fp8_tile_probe(act_packed_3, weight_packed_3)
    out_ref_3 = _scalar_reference(act_nibbles_3, weight_rows_3).to("cuda")
    diff_3 = (out_cuda_3 - out_ref_3).abs()
    max_abs_err_3 = float(diff_3.max().item())
    rel_l2_3 = float((out_cuda_3 - out_ref_3).norm() / out_ref_3.norm().clamp_min(1e-9))

    print(f"[sp12a] test 3 max_abs_err = {max_abs_err_3:.6e}")
    print(f"[sp12a] test 3 rel_l2      = {rel_l2_3:.6e}")
    test3_pass = max_abs_err_3 < 1e-4

    # Timing
    print(f"\n[sp12a] === timing FP8 MMA tile (m=16, n=8, k=32) ===")
    for _ in range(20):
        module.fp8_tile_probe(act_packed_3, weight_packed_3)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    iters = 10000
    for _ in range(iters):
        module.fp8_tile_probe(act_packed_3, weight_packed_3)
    end.record()
    torch.cuda.synchronize()
    per_call_us = float(start.elapsed_time(end) / iters * 1000.0)
    print(f"[sp12a] timing: {per_call_us:.3f} us/call ({iters} iters)")

    summary = {
        "type": "sp12a_sm121_fp8_e2m1_tile_probe",
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(cap),
        "build_seconds": build_seconds,
        "tests": [
            {"name": "positive_const_act_ramp_weight", "max_abs_err": max_abs_err, "rel_l2": rel_l2, "pass": test1_pass},
            {"name": "mixed_signs", "max_abs_err": max_abs_err_2, "rel_l2": rel_l2_2, "pass": test2_pass},
            {"name": "random_nibbles", "max_abs_err": max_abs_err_3, "rel_l2": rel_l2_3, "pass": test3_pass},
        ],
        "per_call_us": per_call_us,
        "all_pass": test1_pass and test2_pass and test3_pass,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n[sp12a] === SUMMARY ===")
    for t in summary["tests"]:
        print(f"[sp12a]   {t['name']:35s}  max_abs_err={t['max_abs_err']:.3e}  {'PASS' if t['pass'] else 'FAIL'}")
    print(f"[sp12a]   per-call: {per_call_us:.3f} us")
    print(f"[sp12a]   ALL PASS: {summary['all_pass']}")
    print(f"[sp12a] report: {out_path}")
    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
