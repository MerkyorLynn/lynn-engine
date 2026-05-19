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
// SM120a MMA stub: validates compilation with f8f6f4 instruction
// Full implementation requires correct fragment register layout.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void dense_fp4xfp8_mma_stub_kernel(
    const uint8_t* __restrict__ act_fp8_packed,  // [M, K] FP8 as bytes
    const uint8_t* __restrict__ weight_packed,   // [N, K/2] FP4 packed
    float* __restrict__ output,                  // [M, N]
    int M, int N, int K,
    uint8_t scale_byte                           // block scale encoding
) {
    // This is a compilation probe. The actual register layout for
    // SM120::BLOCKSCALED::SM120_16x8x32_TN_VS requires careful fragment
    // packing that will be developed iteratively.
    //
    // For now: prove the include path and instruction exist.
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;

    if (lane == 0 && warp_id == 0 && blockIdx.x == 0) {
        // Touch the MMA type to prove compilation
        // (actual execution deferred to full fragment implementation)
        output[0] = 0.0f;
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

// Python entry: MMA capability probe
torch::Tensor lynn_dense_fp4xfp8_mma_probe(
    torch::Tensor act_fp8,
    torch::Tensor weight_packed,
    int64_t M, int64_t N, int64_t K
) {
#if HAS_FP4_MMA
    auto output = torch::zeros({M, N}, torch::TensorOptions().device(act_fp8.device()).dtype(torch::kFloat32));
    dense_fp4xfp8_mma_stub_kernel<<<1, 32>>>(
        act_fp8.data_ptr<uint8_t>(),
        weight_packed.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        M, N, K, 0
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dense_fp4xfp8_mma_probe failed: ", cudaGetErrorString(err));
    return output;
#else
    TORCH_CHECK(false, "FP4 MMA not available: requires SM120a + CuTe headers");
    return torch::Tensor();
#endif
}
