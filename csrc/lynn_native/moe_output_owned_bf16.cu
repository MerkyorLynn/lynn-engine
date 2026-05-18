/**
 * Lynn Engine · Native MoE Output-Owned BF16 Kernel
 *
 * Two-stage BF16 decode MoE for single-token inference:
 *   Stage 1: gate_up_silu — computes silu(gate(x)) * up(x) for each active expert
 *   Stage 2: down_weighted_sum — output-owned reduction over 8 experts (non-atomic)
 *
 * Design: BF16 vectorized loads + warp shuffle reduction.
 * Target: < 0.059ms on RTX PRO 6000 Blackwell (vs Triton active baseline).
 *
 * Weight layout (BF16 fused): same as engine/moe_optimized.py
 *   gate_up_weight: [num_experts, 2*intermediate, hidden] = [256, 1024, 2048] BF16
 *   down_weight:    [num_experts, hidden, intermediate]   = [256, 2048, 512]  BF16
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHidden = 2048;
constexpr int kIntermediate = 512;
constexpr int kGateUpRows = kIntermediate * 2;  // 1024
constexpr int kTopK = 8;

// ─────────────────────────────────────────────────────────────────────────────
// Stage 1: gate_up_silu_bf16_kernel
//
// Grid: (top_k, intermediate / TILE_I)
// Block: THREADS threads
//
// Each block computes TILE_I intermediate outputs for one expert slot.
// Loads x once into shared memory, then for each of TILE_I rows:
//   gate_acc = dot(gate_w[expert, row, :], x)
//   up_acc   = dot(up_w[expert, row, :], x)
//   inter[slot, row] = bf16(silu(gate_acc) * up_acc)
// ─────────────────────────────────────────────────────────────────────────────

template <int TILE_I, int THREADS>
__global__ void gate_up_silu_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,         // [hidden]
    const int32_t* __restrict__ expert_ids,       // [top_k]
    const __nv_bfloat16* __restrict__ gate_up_w,  // [num_experts, 1024, 2048] row-major
    __nv_bfloat16* __restrict__ inter,            // [top_k, intermediate]
    int64_t w_stride_e,   // stride over expert dim (in elements)
    int64_t w_stride_m,   // stride over row dim
    int64_t w_stride_n    // stride over col dim (=1 for contiguous)
) {
    const int slot = blockIdx.x;
    const int tile_start = blockIdx.y * TILE_I;
    const int tid = threadIdx.x;
    const int expert = expert_ids[slot];

    // Load x into shared memory for reuse across TILE_I rows
    __shared__ float x_shared[kHidden];
    for (int h = tid; h < kHidden; h += THREADS) {
        x_shared[h] = __bfloat162float(x[h]);
    }
    __syncthreads();

    // Compute TILE_I gate + up dot products
    float gate_acc[TILE_I];
    float up_acc[TILE_I];
    #pragma unroll
    for (int r = 0; r < TILE_I; ++r) {
        gate_acc[r] = 0.0f;
        up_acc[r] = 0.0f;
    }

    // Tile over hidden dimension — each thread handles stride of THREADS
    for (int h = tid; h < kHidden; h += THREADS) {
        const float xh = x_shared[h];
        #pragma unroll
        for (int r = 0; r < TILE_I; ++r) {
            const int gate_row = tile_start + r;
            const int up_row = kIntermediate + tile_start + r;
            if (gate_row < kIntermediate) {
                const float gw = __bfloat162float(
                    gate_up_w[static_cast<int64_t>(expert) * w_stride_e +
                              static_cast<int64_t>(gate_row) * w_stride_m +
                              static_cast<int64_t>(h) * w_stride_n]);
                const float uw = __bfloat162float(
                    gate_up_w[static_cast<int64_t>(expert) * w_stride_e +
                              static_cast<int64_t>(up_row) * w_stride_m +
                              static_cast<int64_t>(h) * w_stride_n]);
                gate_acc[r] += gw * xh;
                up_acc[r] += uw * xh;
            }
        }
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int r = 0; r < TILE_I; ++r) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc[r] += __shfl_down_sync(0xFFFFFFFF, gate_acc[r], offset);
            up_acc[r] += __shfl_down_sync(0xFFFFFFFF, up_acc[r], offset);
        }
    }

    // Cross-warp reduction via shared memory
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = THREADS / 32;

    __shared__ float warp_gate[TILE_I][32];  // max 32 warps
    __shared__ float warp_up[TILE_I][32];

    if (lane_id == 0) {
        #pragma unroll
        for (int r = 0; r < TILE_I; ++r) {
            warp_gate[r][warp_id] = gate_acc[r];
            warp_up[r][warp_id] = up_acc[r];
        }
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0 && lane_id < num_warps) {
        const unsigned int warp_mask = (1u << num_warps) - 1u;
        #pragma unroll
        for (int r = 0; r < TILE_I; ++r) {
            float g = warp_gate[r][lane_id];
            float u = warp_up[r][lane_id];
            // Warp shuffle to reduce across warps
            #pragma unroll
            for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
                g += __shfl_down_sync(warp_mask, g, offset);
                u += __shfl_down_sync(warp_mask, u, offset);
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

// ─────────────────────────────────────────────────────────────────────────────
// Stage 2: down_weighted_sum_bf16_kernel (OUTPUT-OWNED, NON-ATOMIC)
//
// Grid: (hidden / TILE_H,)
// Block: THREADS threads
//
// Each block owns TILE_H output hidden elements.
// For each expert k in [0..top_k):
//   Load inter[k, 0..512] (vectorized)
//   For each owned output h in [tile_start..tile_start+TILE_H):
//     acc[h] += routing_w[k] * dot(down_w[expert_k, h, :], inter[k, :])
// Final: out[h] = bf16(acc[h])
//
// Key: no atomics needed because each block exclusively owns its output tile.
// ─────────────────────────────────────────────────────────────────────────────

template <int TILE_H, int THREADS>
__global__ void down_weighted_sum_bf16_kernel(
    const __nv_bfloat16* __restrict__ inter,       // [top_k, intermediate]
    const int32_t* __restrict__ expert_ids,        // [top_k]
    const float* __restrict__ routing_weights,     // [top_k]
    const __nv_bfloat16* __restrict__ down_w,      // [num_experts, hidden, intermediate]
    __nv_bfloat16* __restrict__ out,               // [hidden]
    int64_t dw_stride_e,   // stride over expert dim
    int64_t dw_stride_m,   // stride over hidden (row) dim
    int64_t dw_stride_n,   // stride over intermediate (col) dim
    int64_t top_k
) {
    const int tile_start = blockIdx.x * TILE_H;
    const int tid = threadIdx.x;

    // Each thread accumulates TILE_H output values across all experts
    // Strategy: each thread handles a subset of the intermediate reduction
    // then we reduce across threads per output row.

    // Shared memory for intermediate values (reused per expert)
    __shared__ float inter_shared[kIntermediate];

    // Output accumulators — one per tile row
    float out_acc[TILE_H];
    #pragma unroll
    for (int r = 0; r < TILE_H; ++r) {
        out_acc[r] = 0.0f;
    }

    // Loop over all active experts
    for (int k = 0; k < top_k; ++k) {
        const int expert = expert_ids[k];
        // Match the Python slot-order reference:
        //   out += ffn * routing_weights[k].to(torch.bfloat16)
        // Applying the raw FP32 route changes BF16 accumulation semantics.
        const float route_w = __bfloat162float(__float2bfloat16(routing_weights[k]));

        // Load inter[k, :] into shared memory (collaborative)
        for (int i = tid; i < kIntermediate; i += THREADS) {
            inter_shared[i] = __bfloat162float(
                inter[static_cast<int64_t>(k) * kIntermediate + i]);
        }
        __syncthreads();

        // Each thread computes partial dot products for TILE_H output rows
        // Divide intermediate dim across threads: each thread handles
        // kIntermediate/THREADS elements per row
        #pragma unroll
        for (int r = 0; r < TILE_H; ++r) {
            const int h = tile_start + r;
            if (h < kHidden) {
                float dot = 0.0f;
                for (int i = tid; i < kIntermediate; i += THREADS) {
                    const float dw_val = __bfloat162float(
                        down_w[static_cast<int64_t>(expert) * dw_stride_e +
                               static_cast<int64_t>(h) * dw_stride_m +
                               static_cast<int64_t>(i) * dw_stride_n]);
                    dot += dw_val * inter_shared[i];
                }
                out_acc[r] += route_w * dot;
            }
        }
        __syncthreads();  // Ensure inter_shared safe for next expert
    }

    // Reduce out_acc across threads (each thread has partial sums)
    // Use shared memory for cross-thread reduction
    __shared__ float reduce_buf[TILE_H * THREADS];

    #pragma unroll
    for (int r = 0; r < TILE_H; ++r) {
        reduce_buf[r * THREADS + tid] = out_acc[r];
    }
    __syncthreads();

    // Tree reduction
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int r = 0; r < TILE_H; ++r) {
                reduce_buf[r * THREADS + tid] += reduce_buf[r * THREADS + tid + stride];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes final results
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

// ─────────────────────────────────────────────────────────────────────────────
// Python-callable entry points
// ─────────────────────────────────────────────────────────────────────────────

torch::Tensor lynn_native_moe_output_owned_bf16(
    torch::Tensor x,                   // [2048] BF16
    torch::Tensor expert_ids,          // [top_k] int32
    torch::Tensor routing_weights,     // [top_k] float32
    torch::Tensor gate_up_weight,      // [num_experts, 1024, 2048] BF16
    torch::Tensor down_weight          // [num_experts, 2048, 512] BF16
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(x.dim() == 1 && x.numel() == kHidden, "x must be [2048]");
    TORCH_CHECK(expert_ids.is_cuda() && expert_ids.scalar_type() == torch::kInt32);
    TORCH_CHECK(routing_weights.is_cuda() && routing_weights.scalar_type() == torch::kFloat32);
    TORCH_CHECK(gate_up_weight.is_cuda() && gate_up_weight.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(down_weight.is_cuda() && down_weight.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(gate_up_weight.dim() == 3, "gate_up_weight must be [E, 1024, 2048]");
    TORCH_CHECK(gate_up_weight.size(1) == kGateUpRows && gate_up_weight.size(2) == kHidden);
    TORCH_CHECK(down_weight.dim() == 3, "down_weight must be [E, 2048, 512]");
    TORCH_CHECK(down_weight.size(1) == kHidden && down_weight.size(2) == kIntermediate);

    const int top_k = expert_ids.size(0);
    TORCH_CHECK(routing_weights.size(0) == top_k);

    // Allocate intermediate buffer [top_k, 512] BF16
    auto inter = torch::empty({top_k, kIntermediate},
                              torch::TensorOptions().device(x.device()).dtype(torch::kBFloat16));
    // Allocate output [2048] BF16
    auto out = torch::zeros({kHidden},
                            torch::TensorOptions().device(x.device()).dtype(torch::kBFloat16));

    // Stage 1: gate_up_silu
    constexpr int TILE_I = 4;
    constexpr int THREADS_S1 = 256;
    dim3 grid_s1(top_k, (kIntermediate + TILE_I - 1) / TILE_I);
    dim3 block_s1(THREADS_S1);

    gate_up_silu_bf16_kernel<TILE_I, THREADS_S1><<<grid_s1, block_s1>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        expert_ids.data_ptr<int32_t>(),
        reinterpret_cast<const __nv_bfloat16*>(gate_up_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(inter.data_ptr<at::BFloat16>()),
        gate_up_weight.stride(0),
        gate_up_weight.stride(1),
        gate_up_weight.stride(2)
    );

    // Stage 2: down_weighted_sum (output-owned)
    constexpr int TILE_H = 8;
    constexpr int THREADS_S2 = 256;
    dim3 grid_s2((kHidden + TILE_H - 1) / TILE_H);
    dim3 block_s2(THREADS_S2);

    down_weighted_sum_bf16_kernel<TILE_H, THREADS_S2><<<grid_s2, block_s2>>>(
        reinterpret_cast<const __nv_bfloat16*>(inter.data_ptr<at::BFloat16>()),
        expert_ids.data_ptr<int32_t>(),
        routing_weights.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(down_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        down_weight.stride(0),
        down_weight.stride(1),
        down_weight.stride(2),
        top_k
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "lynn_native_moe_output_owned_bf16 failed: ", cudaGetErrorString(err));

    return out;
}

torch::Tensor lynn_native_moe_slot_output_owned_bf16(
    torch::Tensor x,                   // [2048] BF16
    torch::Tensor routing_weights,     // [top_k] float32
    torch::Tensor slot_gate_up,        // [top_k, 1024, 2048] BF16 — pre-gathered
    torch::Tensor slot_down            // [top_k, 2048, 512] BF16 — pre-gathered
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(x.dim() == 1 && x.numel() == kHidden, "x must be [2048]");
    TORCH_CHECK(routing_weights.is_cuda() && routing_weights.scalar_type() == torch::kFloat32);
    TORCH_CHECK(slot_gate_up.is_cuda() && slot_gate_up.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(slot_down.is_cuda() && slot_down.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(slot_gate_up.dim() == 3, "slot_gate_up must be [top_k, 1024, 2048]");
    TORCH_CHECK(slot_gate_up.size(1) == kGateUpRows && slot_gate_up.size(2) == kHidden);
    TORCH_CHECK(slot_down.dim() == 3, "slot_down must be [top_k, 2048, 512]");
    TORCH_CHECK(slot_down.size(1) == kHidden && slot_down.size(2) == kIntermediate);

    const int top_k = slot_gate_up.size(0);
    TORCH_CHECK(routing_weights.size(0) == top_k);
    TORCH_CHECK(slot_down.size(0) == top_k);

    // Slot variant: dim 0 IS the slot index, no expert_ids needed.
    // Reuse the same kernels — gate_up_silu_bf16_kernel uses expert_ids to
    // index into weight[expert, ...], but here slot_gate_up[slot, ...] is
    // already gathered, so we pass a trivial identity mapping: expert_ids = [0,1,...,top_k-1]
    auto identity_ids = torch::arange(top_k, torch::TensorOptions().device(x.device()).dtype(torch::kInt32));

    // Allocate intermediate buffer [top_k, 512] BF16
    auto inter = torch::empty({top_k, kIntermediate},
                              torch::TensorOptions().device(x.device()).dtype(torch::kBFloat16));
    // Allocate output [2048] BF16
    auto out = torch::zeros({kHidden},
                            torch::TensorOptions().device(x.device()).dtype(torch::kBFloat16));

    // Stage 1: gate_up_silu (slot_gate_up[slot, row, col] — identity expert mapping)
    constexpr int TILE_I = 4;
    constexpr int THREADS_S1 = 256;
    dim3 grid_s1(top_k, (kIntermediate + TILE_I - 1) / TILE_I);
    dim3 block_s1(THREADS_S1);

    gate_up_silu_bf16_kernel<TILE_I, THREADS_S1><<<grid_s1, block_s1>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        identity_ids.data_ptr<int32_t>(),
        reinterpret_cast<const __nv_bfloat16*>(slot_gate_up.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(inter.data_ptr<at::BFloat16>()),
        slot_gate_up.stride(0),
        slot_gate_up.stride(1),
        slot_gate_up.stride(2)
    );

    // Stage 2: down_weighted_sum (output-owned, slot_down[slot, h, i])
    constexpr int TILE_H = 8;
    constexpr int THREADS_S2 = 256;
    dim3 grid_s2((kHidden + TILE_H - 1) / TILE_H);
    dim3 block_s2(THREADS_S2);

    down_weighted_sum_bf16_kernel<TILE_H, THREADS_S2><<<grid_s2, block_s2>>>(
        reinterpret_cast<const __nv_bfloat16*>(inter.data_ptr<at::BFloat16>()),
        identity_ids.data_ptr<int32_t>(),
        routing_weights.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(slot_down.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        slot_down.stride(0),
        slot_down.stride(1),
        slot_down.stride(2),
        top_k
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "lynn_native_moe_slot_output_owned_bf16 failed: ", cudaGetErrorString(err));

    return out;
}

torch::Tensor lynn_native_moe_slot_gate_up_inter_bf16(
    torch::Tensor x,                   // [2048] BF16
    torch::Tensor slot_gate_up         // [top_k, 1024, 2048] BF16 — pre-gathered
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(x.dim() == 1 && x.numel() == kHidden, "x must be [2048]");
    TORCH_CHECK(slot_gate_up.is_cuda() && slot_gate_up.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(slot_gate_up.dim() == 3, "slot_gate_up must be [top_k, 1024, 2048]");
    TORCH_CHECK(slot_gate_up.size(1) == kGateUpRows && slot_gate_up.size(2) == kHidden);

    const int top_k = slot_gate_up.size(0);
    auto identity_ids = torch::arange(top_k, torch::TensorOptions().device(x.device()).dtype(torch::kInt32));
    auto inter = torch::empty({top_k, kIntermediate},
                              torch::TensorOptions().device(x.device()).dtype(torch::kBFloat16));

    constexpr int TILE_I = 4;
    constexpr int THREADS_S1 = 256;
    dim3 grid_s1(top_k, (kIntermediate + TILE_I - 1) / TILE_I);
    dim3 block_s1(THREADS_S1);
    gate_up_silu_bf16_kernel<TILE_I, THREADS_S1><<<grid_s1, block_s1>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        identity_ids.data_ptr<int32_t>(),
        reinterpret_cast<const __nv_bfloat16*>(slot_gate_up.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(inter.data_ptr<at::BFloat16>()),
        slot_gate_up.stride(0),
        slot_gate_up.stride(1),
        slot_gate_up.stride(2)
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "lynn_native_moe_slot_gate_up_inter_bf16 failed: ", cudaGetErrorString(err));
    return inter;
}

torch::Tensor lynn_native_moe_slot_down_weighted_sum_bf16(
    torch::Tensor inter,               // [top_k, 512] BF16
    torch::Tensor routing_weights,     // [top_k] float32
    torch::Tensor slot_down            // [top_k, 2048, 512] BF16 — pre-gathered
) {
    TORCH_CHECK(inter.is_cuda() && inter.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(inter.dim() == 2 && inter.size(1) == kIntermediate, "inter must be [top_k, 512]");
    TORCH_CHECK(routing_weights.is_cuda() && routing_weights.scalar_type() == torch::kFloat32);
    TORCH_CHECK(slot_down.is_cuda() && slot_down.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(slot_down.dim() == 3, "slot_down must be [top_k, 2048, 512]");
    TORCH_CHECK(slot_down.size(1) == kHidden && slot_down.size(2) == kIntermediate);

    const int top_k = inter.size(0);
    TORCH_CHECK(routing_weights.size(0) == top_k);
    TORCH_CHECK(slot_down.size(0) == top_k);

    auto identity_ids = torch::arange(top_k, torch::TensorOptions().device(inter.device()).dtype(torch::kInt32));
    auto out = torch::zeros({kHidden},
                            torch::TensorOptions().device(inter.device()).dtype(torch::kBFloat16));

    constexpr int TILE_H = 8;
    constexpr int THREADS_S2 = 256;
    dim3 grid_s2((kHidden + TILE_H - 1) / TILE_H);
    dim3 block_s2(THREADS_S2);
    down_weighted_sum_bf16_kernel<TILE_H, THREADS_S2><<<grid_s2, block_s2>>>(
        reinterpret_cast<const __nv_bfloat16*>(inter.data_ptr<at::BFloat16>()),
        identity_ids.data_ptr<int32_t>(),
        routing_weights.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(slot_down.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        slot_down.stride(0),
        slot_down.stride(1),
        slot_down.stride(2),
        top_k
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "lynn_native_moe_slot_down_weighted_sum_bf16 failed: ", cudaGetErrorString(err));
    return out;
}
