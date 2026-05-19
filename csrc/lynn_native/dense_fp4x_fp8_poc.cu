/**
 * P191: Dense FP4xFP8 CuTe PoC — Qwen3.5-9B FFN
 *
 * Target instruction: mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32
 * A = E4M3 activation (FP8 quantized from BF16 hidden)
 * B = E2M1 packed NVFP4 weight (from NVFP4 checkpoint)
 * C/D = FP32 accumulator → BF16 output
 *
 * For this PoC we validate compilation and basic correctness vs a reference
 * dequant path. The fragment layout for f8f6f4 MMA requires SM120a and the
 * CuTe headers from deep_gemm.
 *
 * Build: requires -arch=sm_120a and LYNN_ENABLE_SM120A_FP4_MMA=1
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// Check for SM120a FP4 MMA support
#if defined(LYNN_ENABLE_SM120A_FP4_MMA) && __has_include(<cute/arch/mma_sm120.hpp>)
#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>
#define HAS_FP4_MMA 1
#else
#define HAS_FP4_MMA 0
#endif

namespace {
constexpr int kHidden9B = 4096;
constexpr int kIntermediate9B = 12288;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fallback: scalar reference (always builds, for comparison baseline)
// ─────────────────────────────────────────────────────────────────────────────

static __device__ float e2m1_decode_nibble(unsigned char nib) {
    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const bool sign = (nib & 0x08) != 0;
    return sign ? -table[nib & 7] : table[nib & 7];
}

// Simple scalar gate/up kernel for reference validation
__global__ void dense_fp4xfp8_scalar_reference_kernel(
    const __nv_fp8_e4m3* __restrict__ act_fp8,  // [1, K] E4M3
    const float* __restrict__ act_scale,         // [K/16] per-16 scale
    const uint8_t* __restrict__ weight_packed,   // [N, K/2] E2M1 packed
    const float* __restrict__ weight_scale,      // [N, K/16] per-16 scale
    const float* __restrict__ weight_global,     // scalar
    float* __restrict__ output,                  // [N]
    int K, int N
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const float inv_global = 1.0f / weight_global[0];
    float acc = 0.0f;

    for (int k = 0; k < K; ++k) {
        // Decode activation FP8 E4M3
        float a_val = float(act_fp8[k]);
        float a_scale_val = act_scale[k / 16];
        float a = a_val * a_scale_val;

        // Decode weight NVFP4 E2M1
        int packed_col = k / 2;
        bool is_low = (k & 1) == 0;
        uint8_t byte = weight_packed[row * (K / 2) + packed_col];
        unsigned char nib = is_low ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        float w_scale_val = weight_scale[row * (K / 16) + k / 16];
        float w = e2m1_decode_nibble(nib) * w_scale_val * inv_global;

        acc += a * w;
    }
    output[row] = acc;
}

#if HAS_FP4_MMA
// ─────────────────────────────────────────────────────────────────────────────
// SM120a REAL MMA kernel: E4M3 activation × E2M1 weight
//
// Uses SM120_16x8x32_TN<float_e4m3_t, float_e2m1_t, float>::fma
// Shape per MMA: M=16, N=8, K=32
//
// Fragment registers per thread (warp of 32):
//   A (E4M3, row-major): uint32_t[4] — 16 bytes = 16 E4M3 elements
//   B (E2M1, col-major): uint32_t[2] — 8 bytes = 16 E2M1 nibbles
//   D (FP32): float[4]
//
// This kernel computes a single tile for validation. Production would loop
// over K and use multiple tiles for M/N coverage.
//
// Grid: (N/8, M/16) for full matrix; here single-warp single-tile probe.
// ─────────────────────────────────────────────────────────────────────────────

// Fill A fragment: activation E4M3 bytes into registers.
// For M16×K32 with 32 lanes: each lane owns 16 bytes (4 uint32_t).
// NVIDIA WMMA convention for e4m3 in m16n8k32:
//   Thread t owns rows [t%16] and columns based on register index.
//   Simple assumption: thread t holds elements at positions related to
//   its row and K-tile offset. We use a direct sequential fill first.
__device__ __forceinline__ void fill_a_fp8_fragment(
    const uint8_t* act,   // [K] E4M3 bytes, K>=32
    int k_offset,
    int lane,
    uint32_t* a  // [4]
) {
    // P87/P88-proven CuTe fragment layout for SM120 m16n8k32 TN.
    // We broadcast the single decode-token activation across all 16 logical M rows.
    a[0] = 0u;
    a[1] = 0u;
    a[2] = 0u;
    a[3] = 0u;
    const int t0 = lane & 3;
    const int t1 = lane >> 2;
    for (int i = 0; i < 4; ++i) {
        a[i] = 0u;
    }
#pragma unroll
    for (int v = 0; v < 16; ++v) {
        const int v0 = v & 3;
        const int v1 = (v >> 2) & 1;
        const int v2 = (v >> 3) & 1;
        const int offset = t0 * 64 + t1 + v0 * 16 + v1 * 8 + v2 * 256;
        const int k = offset >> 4;
        const uint8_t byte = act[k_offset + k];
        a[v >> 2] |= static_cast<uint32_t>(byte) << (8 * (v & 3));
    }
}

// Fill B fragment: weight E2M1 nibbles (4-bit) into registers.
// For K32×N8 with 32 lanes: each lane owns 2 uint32 = 8 bytes.
// E2M1 is 4-bit, so 8 bytes = 16 nibbles. Total across warp = 16*32 = 512 nibbles = 256 values.
// K32 * N8 = 256 values. Checks out.
__device__ __forceinline__ void fill_b_fp4_fragment(
    const uint8_t* weight,  // [N, K/2] packed nibbles, row n at offset n*(K/2)
    int n_offset,           // which 8-column tile
    int k_offset,           // K offset
    int stride_n,           // K/2 (stride per row in bytes)
    int lane,
    uint32_t* b  // [2]
) {
    b[0] = 0;
    b[1] = 0;
    const int t0 = lane & 3;
    const int t1 = lane >> 2;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const int v0 = v & 3;
        const int v1 = (v >> 2) & 1;
        const int offset = t0 * 32 + t1 + v0 * 8 + v1 * 128;
        const int n = offset & 7;
        const int k = offset >> 3;
        const int byte_offset = (n_offset + n) * stride_n + (k_offset + k) / 2;
        uint8_t byte_val = weight[byte_offset];
        uint8_t nibble = ((k_offset + k) & 1) == 0 ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
        b[v >> 2] |= static_cast<uint32_t>((nibble & 0x0F) << 2) << (8 * (v & 3));
    }
}

__device__ __forceinline__ void fill_a_fp8_fragment_group(
    const uint8_t* act,
    int k_offset,
    int group16,
    int lane,
    uint32_t* a
) {
    a[0] = 0u;
    a[1] = 0u;
    a[2] = 0u;
    a[3] = 0u;
    const int t0 = lane & 3;
    const int t1 = lane >> 2;
#pragma unroll
    for (int v = 0; v < 16; ++v) {
        const int v0 = v & 3;
        const int v1 = (v >> 2) & 1;
        const int v2 = (v >> 3) & 1;
        const int offset = t0 * 64 + t1 + v0 * 16 + v1 * 8 + v2 * 256;
        const int k = offset >> 4;
        const uint8_t byte = ((k >> 4) == group16) ? act[k_offset + k] : 0u;
        a[v >> 2] |= static_cast<uint32_t>(byte) << (8 * (v & 3));
    }
}

__device__ __forceinline__ void fill_b_fp4_fragment_group(
    const uint8_t* weight,
    int n_offset,
    int k_offset,
    int stride_n,
    int group16,
    int lane,
    uint32_t* b
) {
    b[0] = 0u;
    b[1] = 0u;
    const int t0 = lane & 3;
    const int t1 = lane >> 2;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const int v0 = v & 3;
        const int v1 = (v >> 2) & 1;
        const int offset = t0 * 32 + t1 + v0 * 8 + v1 * 128;
        const int n = offset & 7;
        const int k = offset >> 3;
        uint8_t nibble = 0u;
        if ((k >> 4) == group16) {
            const int byte_offset = (n_offset + n) * stride_n + (k_offset + k) / 2;
            const uint8_t byte_val = weight[byte_offset];
            nibble = ((k_offset + k) & 1) == 0 ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
        }
        b[v >> 2] |= static_cast<uint32_t>((nibble & 0x0F) << 2) << (8 * (v & 3));
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

__global__ void dense_fp4xfp8_mma_scaled_kernel(
    const uint8_t* __restrict__ act_fp8,       // [K] E4M3 bytes
    const float* __restrict__ act_scale,       // [K/16]
    const uint8_t* __restrict__ weight_fp4,    // [N, K/2]
    const float* __restrict__ weight_scale,    // [N, K/16]
    const float* __restrict__ weight_global,   // scalar
    float* __restrict__ output,                // [N]
    int N, int K
) {
    const int lane = threadIdx.x & 31;
    const int n_tile = blockIdx.x;
    const int n_offset = n_tile * 8;
    if (n_offset >= N) return;

    const int scale_stride_n = K / 16;
    const float inv_global = 1.0f / weight_global[0];
    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k0 = 0; k0 < K; k0 += 32) {
        float p0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float p1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        uint32_t a0[4];
        uint32_t b0[2];
        uint32_t a1[4];
        uint32_t b1[2];
        fill_a_fp8_fragment_group(act_fp8, k0, 0, lane, a0);
        fill_b_fp4_fragment_group(weight_fp4, n_offset, k0, K / 2, 0, lane, b0);
        fill_a_fp8_fragment_group(act_fp8, k0, 1, lane, a1);
        fill_b_fp4_fragment_group(weight_fp4, n_offset, k0, K / 2, 1, lane, b1);

        cute::SM120_16x8x32_TN<
            cute::float_e4m3_t,
            cute::float_e2m1_t,
            float>::fma(
                p0[0], p0[1], p0[2], p0[3],
                a0[0], a0[1], a0[2], a0[3],
                b0[0], b0[1],
                p0[0], p0[1], p0[2], p0[3]);
        cute::SM120_16x8x32_TN<
            cute::float_e4m3_t,
            cute::float_e2m1_t,
            float>::fma(
                p1[0], p1[1], p1[2], p1[3],
                a1[0], a1[1], a1[2], a1[3],
                b1[0], b1[1],
                p1[0], p1[1], p1[2], p1[3]);

        const int g0 = k0 >> 4;
        const int g1 = g0 + 1;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            int m = 0;
            int n = 0;
            c_coord_from_lane_value(lane, v, &m, &n);
            if (m == 0 && n_offset + n < N) {
                const int row = n_offset + n;
                const float s0 = act_scale[g0] * weight_scale[row * scale_stride_n + g0] * inv_global;
                const float s1 = act_scale[g1] * weight_scale[row * scale_stride_n + g1] * inv_global;
                d[v] += p0[v] * s0 + p1[v] * s1;
            }
        }
    }

#pragma unroll
    for (int v = 0; v < 4; ++v) {
        int m = 0;
        int n = 0;
        c_coord_from_lane_value(lane, v, &m, &n);
        if (m == 0 && n_offset + n < N) {
            output[n_offset + n] = d[v];
        }
    }
}

__global__ void dense_fp4xfp8_mma_real_kernel(
    const uint8_t* __restrict__ act_fp8,     // [K] E4M3 bytes
    const uint8_t* __restrict__ weight_fp4,  // [N, K/2] E2M1 packed
    float* __restrict__ output,              // [N] FP32 output
    int N, int K
) {
    // Single-warp, single K-tile (K=32) probe for first N=8 outputs
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;
    const int n_tile = blockIdx.x;  // which 8-column tile
    const int n_offset = n_tile * 8;

    if (n_offset >= N) return;

    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Loop over K in chunks of 32
    for (int k0 = 0; k0 < K; k0 += 32) {
        uint32_t a[4];
        uint32_t b[2];
        fill_a_fp8_fragment(act_fp8, k0, lane, a);
        fill_b_fp4_fragment(weight_fp4, n_offset, k0, K / 2, lane, b);

        // Execute MMA: accumulate into d
        cute::SM120_16x8x32_TN<
            cute::float_e4m3_t,
            cute::float_e2m1_t,
            float>::fma(
                d[0], d[1], d[2], d[3],
                a[0], a[1], a[2], a[3],
                b[0], b[1],
                d[0], d[1], d[2], d[3]);
    }

    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        int m = 0;
        int n = 0;
        c_coord_from_lane_value(lane, v, &m, &n);
        if (m == 0 && n_offset + n < N) {
            output[n_offset + n] = d[v];
        }
    }
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Python entry: scalar reference path (always available)
// ─────────────────────────────────────────────────────────────────────────────

torch::Tensor lynn_dense_fp4xfp8_scalar_reference(
    torch::Tensor act_fp8,          // [K] or [1,K] uint8 (E4M3 bytes)
    torch::Tensor act_scale,        // [K/16] float32
    torch::Tensor weight_packed,    // [N, K/2] uint8
    torch::Tensor weight_scale,     // [N, K/16] float32
    torch::Tensor weight_global     // scalar float32
) {
    TORCH_CHECK(act_fp8.is_cuda() && act_fp8.scalar_type() == torch::kUInt8);
    TORCH_CHECK(weight_packed.is_cuda() && weight_packed.scalar_type() == torch::kUInt8);

    auto act_flat = act_fp8.dim() == 2 ? act_fp8.view(-1) : act_fp8;
    int K = act_flat.numel();
    int N = weight_packed.size(0);

    TORCH_CHECK(weight_packed.size(1) == K / 2, "weight_packed cols must be K/2");
    TORCH_CHECK(act_scale.numel() == K / 16, "act_scale must be [K/16]");
    TORCH_CHECK(weight_scale.size(0) == N && weight_scale.size(1) == K / 16);

    auto output = torch::zeros({N}, torch::TensorOptions().device(act_fp8.device()).dtype(torch::kFloat32));

    constexpr int THREADS = 256;
    dim3 grid((N + THREADS - 1) / THREADS);

    dense_fp4xfp8_scalar_reference_kernel<<<grid, THREADS>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(act_flat.data_ptr<uint8_t>()),
        act_scale.data_ptr<float>(),
        weight_packed.data_ptr<uint8_t>(),
        weight_scale.data_ptr<float>(),
        weight_global.data_ptr<float>(),
        output.data_ptr<float>(),
        K, N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dense_fp4xfp8_scalar_reference failed: ", cudaGetErrorString(err));
    return output;
}

// Python entry: MMA capability probe — now runs REAL compute
torch::Tensor lynn_dense_fp4xfp8_mma_probe(
    torch::Tensor act_fp8,
    torch::Tensor weight_packed,
    int64_t M, int64_t N, int64_t K
) {
#if HAS_FP4_MMA
    TORCH_CHECK(act_fp8.is_cuda() && act_fp8.scalar_type() == torch::kUInt8);
    TORCH_CHECK(weight_packed.is_cuda() && weight_packed.scalar_type() == torch::kUInt8);
    TORCH_CHECK(K % 32 == 0, "K must be multiple of 32 for MMA tile");

    auto output = torch::zeros({N}, torch::TensorOptions().device(act_fp8.device()).dtype(torch::kFloat32));

    // Launch: one warp per N-tile of 8
    dim3 grid((N + 7) / 8);
    dim3 block(32);  // one warp

    dense_fp4xfp8_mma_real_kernel<<<grid, block>>>(
        act_fp8.data_ptr<uint8_t>(),
        weight_packed.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        N, K
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dense_fp4xfp8_mma_real_kernel failed: ", cudaGetErrorString(err));
    return output;
#else
    TORCH_CHECK(false, "FP4 MMA not available: requires SM120a + CuTe headers");
    return torch::Tensor();
#endif
}

torch::Tensor lynn_dense_fp4xfp8_mma_scaled_probe(
    torch::Tensor act_fp8,
    torch::Tensor act_scale,
    torch::Tensor weight_packed,
    torch::Tensor weight_scale,
    torch::Tensor weight_global,
    int64_t M, int64_t N, int64_t K
) {
#if HAS_FP4_MMA
    TORCH_CHECK(act_fp8.is_cuda() && act_fp8.scalar_type() == torch::kUInt8);
    TORCH_CHECK(act_scale.is_cuda() && act_scale.scalar_type() == torch::kFloat32);
    TORCH_CHECK(weight_packed.is_cuda() && weight_packed.scalar_type() == torch::kUInt8);
    TORCH_CHECK(weight_scale.is_cuda() && weight_scale.scalar_type() == torch::kFloat32);
    TORCH_CHECK(weight_global.is_cuda() && weight_global.scalar_type() == torch::kFloat32);
    TORCH_CHECK(K % 32 == 0, "K must be multiple of 32 for MMA tile");
    TORCH_CHECK(act_scale.numel() == K / 16, "act_scale must be [K/16]");
    TORCH_CHECK(weight_scale.size(0) == N && weight_scale.size(1) == K / 16,
                "weight_scale must be [N, K/16]");

    auto output = torch::zeros({N}, torch::TensorOptions().device(act_fp8.device()).dtype(torch::kFloat32));
    dim3 grid((N + 7) / 8);
    dim3 block(32);
    dense_fp4xfp8_mma_scaled_kernel<<<grid, block>>>(
        act_fp8.data_ptr<uint8_t>(),
        act_scale.data_ptr<float>(),
        weight_packed.data_ptr<uint8_t>(),
        weight_scale.data_ptr<float>(),
        weight_global.data_ptr<float>(),
        output.data_ptr<float>(),
        N, K);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dense_fp4xfp8_mma_scaled_kernel failed: ", cudaGetErrorString(err));
    return output;
#else
    TORCH_CHECK(false, "FP4 MMA not available: requires SM120a + CuTe headers");
    return torch::Tensor();
#endif
}
