/**
 * Lynn Engine · MoE Slot Packed NVFP4 Probe (P140)
 *
 * Correctness-first native CUDA kernel consuming packed NVFP4 slot weights
 * directly. No BF16 dequant pre-pass — E2M1 decode + per-16 scale in kernel.
 *
 * Layout (from p138):
 *   slot_gate_up_packed: [8, 1024, 1024] uint8 (2 nibbles per byte, hidden/2=1024)
 *   slot_gate_up_scale:  [8, 1024, 128]  fp16  (per-16 group, hidden/16=128)
 *   slot_gate_up_global_scale: scalar fp16
 *   slot_down_packed:    [8, 2048, 256]  uint8  (inter/2=256)
 *   slot_down_scale:     [8, 2048, 32]   fp16   (inter/16=32)
 *   slot_down_global_scale: scalar fp16
 *
 * Design:
 *   Stage 1 (gate_up): each block computes TILE_I intermediate values for one slot.
 *     For each h in [0..2048): decode NVFP4 nibble, multiply scale/global, FMA with x[h].
 *     Warp shuffle reduction → silu(gate) * up → inter[slot, row] BF16
 *
 *   Stage 2 (down_weighted_sum): output-owned. Each block owns TILE_H output elements.
 *     For each slot k: load inter[k, i], decode down weight nibble, FMA, accumulate.
 *     Final: out[h] = sum_k(route_w_k * dot(down[k,h,:], inter[k,:]))
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHidden = 2048;
constexpr int kIntermediate = 512;
constexpr int kGateUpRows = 1024;  // 512 gate + 512 up
constexpr int kGroupSize = 16;

// E2M1 FP4 decode table
__device__ __forceinline__ float e2m1_decode(unsigned char nibble) {
    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const unsigned char mag = nibble & 0x07;
    const bool sign = (nibble & 0x08) != 0;
    return sign ? -table[mag] : table[mag];
}

// ── Stage 1: gate_up packed NVFP4 kernel ──
// Grid: (top_k, intermediate / TILE_I)
// Block: THREADS
template <int TILE_I, int THREADS>
__global__ void gate_up_packed_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ x,            // [hidden]
    const uint8_t* __restrict__ gate_up_packed,     // [top_k, 1024, 1024]
    const __half* __restrict__ gate_up_scale,       // [top_k, 1024, 128]
    const __half* __restrict__ gate_up_global_scale,// scalar
    __nv_bfloat16* __restrict__ inter,              // [top_k, 512]
    int64_t packed_stride_k, int64_t packed_stride_r, int64_t packed_stride_c,
    int64_t scale_stride_k, int64_t scale_stride_r, int64_t scale_stride_g
) {
    const int slot = blockIdx.x;
    const int tile_start = blockIdx.y * TILE_I;
    const int tid = threadIdx.x;
    const float inv_global = 1.0f / __half2float(gate_up_global_scale[0]);

    // Load x into shared memory
    __shared__ float x_shared[kHidden];
    for (int h = tid; h < kHidden; h += THREADS) {
        x_shared[h] = __bfloat162float(x[h]);
    }
    __syncthreads();

    float gate_acc[TILE_I];
    float up_acc[TILE_I];
    #pragma unroll
    for (int r = 0; r < TILE_I; ++r) {
        gate_acc[r] = 0.0f;
        up_acc[r] = 0.0f;
    }

    // Iterate over hidden dim
    for (int h = tid; h < kHidden; h += THREADS) {
        const float xh = x_shared[h];
        const int packed_col = h >> 1;       // h/2
        const int scale_group = h >> 4;      // h/16
        const bool is_low = (h & 1) == 0;

        #pragma unroll
        for (int r = 0; r < TILE_I; ++r) {
            const int gate_row = tile_start + r;
            const int up_row = kIntermediate + tile_start + r;
            if (gate_row < kIntermediate) {
                // Gate weight
                const uint8_t g_byte = gate_up_packed[
                    slot * packed_stride_k + gate_row * packed_stride_r + packed_col * packed_stride_c];
                const unsigned char g_nib = is_low ? (g_byte & 0x0F) : ((g_byte >> 4) & 0x0F);
                const float g_scale = __half2float(gate_up_scale[
                    slot * scale_stride_k + gate_row * scale_stride_r + scale_group * scale_stride_g]);
                gate_acc[r] += e2m1_decode(g_nib) * g_scale * inv_global * xh;

                // Up weight
                const uint8_t u_byte = gate_up_packed[
                    slot * packed_stride_k + up_row * packed_stride_r + packed_col * packed_stride_c];
                const unsigned char u_nib = is_low ? (u_byte & 0x0F) : ((u_byte >> 4) & 0x0F);
                const float u_scale = __half2float(gate_up_scale[
                    slot * scale_stride_k + up_row * scale_stride_r + scale_group * scale_stride_g]);
                up_acc[r] += e2m1_decode(u_nib) * u_scale * inv_global * xh;
            }
        }
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int r = 0; r < TILE_I; ++r) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc[r] += __shfl_down_sync(0xFFFFFFFF, gate_acc[r], offset);
            up_acc[r] += __shfl_down_sync(0xFFFFFFFF, up_acc[r], offset);
        }
    }

    // Cross-warp reduction
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = THREADS / 32;

    __shared__ float warp_gate[TILE_I][32];
    __shared__ float warp_up[TILE_I][32];

    if (lane_id == 0) {
        #pragma unroll
        for (int r = 0; r < TILE_I; ++r) {
            warp_gate[r][warp_id] = gate_acc[r];
            warp_up[r][warp_id] = up_acc[r];
        }
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < num_warps) {
        #pragma unroll
        for (int r = 0; r < TILE_I; ++r) {
            float g = warp_gate[r][lane_id];
            float u = warp_up[r][lane_id];
            for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
                g += __shfl_down_sync(0xFFFFFFFF, g, offset);
                u += __shfl_down_sync(0xFFFFFFFF, u, offset);
            }
            if (lane_id == 0) {
                const int row = tile_start + r;
                if (row < kIntermediate) {
                    const float silu_g = g / (1.0f + expf(-g));
                    inter[static_cast<int64_t>(slot) * kIntermediate + row] =
                        __float2bfloat16(silu_g * u);
                }
            }
        }
    }
}

// ── Stage 2: down weighted sum packed NVFP4 kernel (output-owned) ──
// Grid: (hidden / TILE_H,)
// Block: THREADS
template <int TILE_H, int THREADS>
__global__ void down_weighted_sum_packed_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ inter,         // [top_k, 512]
    const float* __restrict__ routing_weights,       // [top_k]
    const uint8_t* __restrict__ down_packed,         // [top_k, 2048, 256]
    const __half* __restrict__ down_scale,           // [top_k, 2048, 32]
    const __half* __restrict__ down_global_scale,    // scalar
    __nv_bfloat16* __restrict__ out,                 // [hidden]
    int64_t packed_stride_k, int64_t packed_stride_r, int64_t packed_stride_c,
    int64_t scale_stride_k, int64_t scale_stride_r, int64_t scale_stride_g,
    int64_t top_k
) {
    const int tile_start = blockIdx.x * TILE_H;
    const int tid = threadIdx.x;
    const float inv_global = 1.0f / __half2float(down_global_scale[0]);

    // Shared memory for inter values (reused per slot)
    __shared__ float inter_shared[kIntermediate];

    float out_acc[TILE_H];
    #pragma unroll
    for (int r = 0; r < TILE_H; ++r) {
        out_acc[r] = 0.0f;
    }

    for (int k = 0; k < top_k; ++k) {
        const float route_w = routing_weights[k];

        // Load inter[k, :] into shared memory
        for (int i = tid; i < kIntermediate; i += THREADS) {
            inter_shared[i] = __bfloat162float(inter[k * kIntermediate + i]);
        }
        __syncthreads();

        // Each thread computes partial dot for TILE_H output rows
        #pragma unroll
        for (int r = 0; r < TILE_H; ++r) {
            const int h = tile_start + r;
            if (h < kHidden) {
                float dot = 0.0f;
                for (int i = tid; i < kIntermediate; i += THREADS) {
                    const int packed_col = i >> 1;
                    const int scale_group = i >> 4;
                    const bool is_low = (i & 1) == 0;

                    const uint8_t byte = down_packed[
                        k * packed_stride_k + h * packed_stride_r + packed_col * packed_stride_c];
                    const unsigned char nib = is_low ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                    const float scale = __half2float(down_scale[
                        k * scale_stride_k + h * scale_stride_r + scale_group * scale_stride_g]);
                    dot += e2m1_decode(nib) * scale * inv_global * inter_shared[i];
                }
                out_acc[r] += route_w * dot;
            }
        }
        __syncthreads();
    }

    // Reduce across threads
    __shared__ float reduce_buf[TILE_H * THREADS];
    #pragma unroll
    for (int r = 0; r < TILE_H; ++r) {
        reduce_buf[r * THREADS + tid] = out_acc[r];
    }
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int r = 0; r < TILE_H; ++r) {
                reduce_buf[r * THREADS + tid] += reduce_buf[r * THREADS + tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < TILE_H; ++r) {
            const int h = tile_start + r;
            if (h < kHidden) {
                out[h] = __float2bfloat16(reduce_buf[r * THREADS]);
            }
        }
    }
}

}  // namespace

// ─── Python entry point ───

torch::Tensor lynn_native_moe_slot_packed_nvfp4_inter_probe(
    torch::Tensor x,                        // [2048] BF16
    torch::Tensor slot_gate_up_packed,      // [top_k, 1024, 1024] uint8
    torch::Tensor slot_gate_up_scale,       // [top_k, 1024, 128] fp16
    torch::Tensor slot_gate_up_global_scale // scalar fp16
) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(x.dim() == 1 && x.numel() == kHidden);
    TORCH_CHECK(slot_gate_up_packed.is_cuda() && slot_gate_up_packed.scalar_type() == torch::kUInt8);
    TORCH_CHECK(slot_gate_up_scale.is_cuda() && slot_gate_up_scale.scalar_type() == torch::kFloat16);
    TORCH_CHECK(slot_gate_up_global_scale.is_cuda() && slot_gate_up_global_scale.scalar_type() == torch::kFloat16);
    const int top_k = slot_gate_up_packed.size(0);
    TORCH_CHECK(slot_gate_up_packed.size(1) == kGateUpRows && slot_gate_up_packed.size(2) == kHidden / 2);
    TORCH_CHECK(slot_gate_up_scale.size(0) == top_k && slot_gate_up_scale.size(1) == kGateUpRows);
    TORCH_CHECK(slot_gate_up_scale.size(2) == kHidden / kGroupSize);

    auto inter = torch::empty({top_k, kIntermediate},
        torch::TensorOptions().device(x.device()).dtype(torch::kBFloat16));

    constexpr int TILE_I = 4;
    constexpr int THREADS_S1 = 256;
    dim3 grid_s1(top_k, (kIntermediate + TILE_I - 1) / TILE_I);
    gate_up_packed_nvfp4_kernel<TILE_I, THREADS_S1><<<grid_s1, THREADS_S1>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        slot_gate_up_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(slot_gate_up_scale.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(slot_gate_up_global_scale.data_ptr<at::Half>()),
        reinterpret_cast<__nv_bfloat16*>(inter.data_ptr<at::BFloat16>()),
        slot_gate_up_packed.stride(0), slot_gate_up_packed.stride(1), slot_gate_up_packed.stride(2),
        slot_gate_up_scale.stride(0), slot_gate_up_scale.stride(1), slot_gate_up_scale.stride(2)
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "moe_slot_packed_nvfp4_inter_probe failed: ", cudaGetErrorString(err));
    return inter;
}

torch::Tensor lynn_native_moe_slot_packed_nvfp4_down_probe(
    torch::Tensor inter,                    // [top_k, 512] BF16
    torch::Tensor routing_weights,          // [top_k] float32
    torch::Tensor slot_down_packed,         // [top_k, 2048, 256] uint8
    torch::Tensor slot_down_scale,          // [top_k, 2048, 32] fp16
    torch::Tensor slot_down_global_scale    // scalar fp16
) {
    TORCH_CHECK(inter.is_cuda() && inter.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(inter.dim() == 2 && inter.size(1) == kIntermediate);
    TORCH_CHECK(routing_weights.is_cuda() && routing_weights.scalar_type() == torch::kFloat32);
    TORCH_CHECK(slot_down_packed.is_cuda() && slot_down_packed.scalar_type() == torch::kUInt8);
    TORCH_CHECK(slot_down_scale.is_cuda() && slot_down_scale.scalar_type() == torch::kFloat16);
    TORCH_CHECK(slot_down_global_scale.is_cuda() && slot_down_global_scale.scalar_type() == torch::kFloat16);
    const int top_k = inter.size(0);
    TORCH_CHECK(routing_weights.size(0) == top_k);
    TORCH_CHECK(slot_down_packed.size(0) == top_k && slot_down_packed.size(1) == kHidden);
    TORCH_CHECK(slot_down_packed.size(2) == kIntermediate / 2);
    TORCH_CHECK(slot_down_scale.size(0) == top_k && slot_down_scale.size(1) == kHidden);
    TORCH_CHECK(slot_down_scale.size(2) == kIntermediate / kGroupSize);

    auto out = torch::zeros({kHidden},
        torch::TensorOptions().device(inter.device()).dtype(torch::kBFloat16));

    constexpr int TILE_H = 8;
    constexpr int THREADS_S2 = 256;
    dim3 grid_s2((kHidden + TILE_H - 1) / TILE_H);
    down_weighted_sum_packed_nvfp4_kernel<TILE_H, THREADS_S2><<<grid_s2, THREADS_S2>>>(
        reinterpret_cast<const __nv_bfloat16*>(inter.data_ptr<at::BFloat16>()),
        routing_weights.data_ptr<float>(),
        slot_down_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(slot_down_scale.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(slot_down_global_scale.data_ptr<at::Half>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        slot_down_packed.stride(0), slot_down_packed.stride(1), slot_down_packed.stride(2),
        slot_down_scale.stride(0), slot_down_scale.stride(1), slot_down_scale.stride(2),
        top_k
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "moe_slot_packed_nvfp4_down_probe failed: ", cudaGetErrorString(err));
    return out;
}

torch::Tensor lynn_native_moe_slot_packed_nvfp4_probe(
    torch::Tensor x,                        // [2048] BF16
    torch::Tensor routing_weights,          // [top_k] float32
    torch::Tensor slot_gate_up_packed,      // [top_k, 1024, 1024] uint8
    torch::Tensor slot_gate_up_scale,       // [top_k, 1024, 128] fp16
    torch::Tensor slot_gate_up_global_scale,// scalar fp16
    torch::Tensor slot_down_packed,         // [top_k, 2048, 256] uint8
    torch::Tensor slot_down_scale,          // [top_k, 2048, 32] fp16
    torch::Tensor slot_down_global_scale    // scalar fp16
) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(x.dim() == 1 && x.numel() == kHidden);
    TORCH_CHECK(routing_weights.is_cuda() && routing_weights.scalar_type() == torch::kFloat32);
    TORCH_CHECK(slot_gate_up_packed.is_cuda() && slot_gate_up_packed.scalar_type() == torch::kUInt8);
    TORCH_CHECK(slot_gate_up_scale.is_cuda() && slot_gate_up_scale.scalar_type() == torch::kFloat16);
    TORCH_CHECK(slot_gate_up_global_scale.is_cuda() && slot_gate_up_global_scale.scalar_type() == torch::kFloat16);
    TORCH_CHECK(slot_down_packed.is_cuda() && slot_down_packed.scalar_type() == torch::kUInt8);
    TORCH_CHECK(slot_down_scale.is_cuda() && slot_down_scale.scalar_type() == torch::kFloat16);
    TORCH_CHECK(slot_down_global_scale.is_cuda() && slot_down_global_scale.scalar_type() == torch::kFloat16);

    const int top_k = slot_gate_up_packed.size(0);
    TORCH_CHECK(routing_weights.size(0) == top_k);
    TORCH_CHECK(slot_gate_up_packed.size(1) == kGateUpRows && slot_gate_up_packed.size(2) == kHidden / 2);
    TORCH_CHECK(slot_gate_up_scale.size(1) == kGateUpRows && slot_gate_up_scale.size(2) == kHidden / kGroupSize);
    TORCH_CHECK(slot_down_packed.size(0) == top_k && slot_down_packed.size(1) == kHidden && slot_down_packed.size(2) == kIntermediate / 2);
    TORCH_CHECK(slot_down_scale.size(0) == top_k && slot_down_scale.size(1) == kHidden && slot_down_scale.size(2) == kIntermediate / kGroupSize);

    // Allocate inter [top_k, 512] BF16
    auto inter = torch::empty({top_k, kIntermediate},
        torch::TensorOptions().device(x.device()).dtype(torch::kBFloat16));
    // Allocate output [2048] BF16
    auto out = torch::zeros({kHidden},
        torch::TensorOptions().device(x.device()).dtype(torch::kBFloat16));

    // Stage 1: gate_up
    constexpr int TILE_I = 4;
    constexpr int THREADS_S1 = 256;
    dim3 grid_s1(top_k, (kIntermediate + TILE_I - 1) / TILE_I);

    gate_up_packed_nvfp4_kernel<TILE_I, THREADS_S1><<<grid_s1, THREADS_S1>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        slot_gate_up_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(slot_gate_up_scale.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(slot_gate_up_global_scale.data_ptr<at::Half>()),
        reinterpret_cast<__nv_bfloat16*>(inter.data_ptr<at::BFloat16>()),
        slot_gate_up_packed.stride(0), slot_gate_up_packed.stride(1), slot_gate_up_packed.stride(2),
        slot_gate_up_scale.stride(0), slot_gate_up_scale.stride(1), slot_gate_up_scale.stride(2)
    );

    // Stage 2: down_weighted_sum
    constexpr int TILE_H = 8;
    constexpr int THREADS_S2 = 256;
    dim3 grid_s2((kHidden + TILE_H - 1) / TILE_H);

    down_weighted_sum_packed_nvfp4_kernel<TILE_H, THREADS_S2><<<grid_s2, THREADS_S2>>>(
        reinterpret_cast<const __nv_bfloat16*>(inter.data_ptr<at::BFloat16>()),
        routing_weights.data_ptr<float>(),
        slot_down_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(slot_down_scale.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(slot_down_global_scale.data_ptr<at::Half>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        slot_down_packed.stride(0), slot_down_packed.stride(1), slot_down_packed.stride(2),
        slot_down_scale.stride(0), slot_down_scale.stride(1), slot_down_scale.stride(2),
        top_k
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "moe_slot_packed_nvfp4_probe failed: ", cudaGetErrorString(err));

    return out;
}
