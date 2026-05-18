/**
 * Lynn Engine · P142 Graph-Safe Pretransposed MoE V3.1
 *
 * TRUE caller-owned hot path. ZERO torch tensor allocation inside.
 * Safe for CUDA graph capture — every buffer is preallocated by caller.
 *
 * Hot path ops (all allocation-free):
 *   1. mm_out(gate_up_scratch, x, W_fused_T)
 *   2. silu_out(inter_scratch, gate_view) + inter_scratch.mul_(up_view)
 *   3. bmm_out(down_scratch, inter_scratch, W_down_T)
 *   4. Custom CUDA kernel: weighted_reduce_bf16 (no torch:: ops)
 *
 * The reduce kernel matches Python semantics:
 *   for each h in [0..2048):
 *     acc = 0.0f (FP32)
 *     for k in [0..8):
 *       acc += bf16(routing_weights_fp32[k]) * bf16_to_fp32(down_scratch[k,0,h])
 *     out[h] = fp32_to_bf16(acc)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
constexpr int kHidden = 2048;
constexpr int kIntermediate = 512;
constexpr int kGateUpFused = 8192;
constexpr int kTopK = 8;

// ─────────────────────────────────────────────────────────────────────────────
// Custom reduce kernel: weighted sum over 8 experts, output-owned per thread.
//
// Grid: (ceil(kHidden / BLOCK_SIZE),)
// Block: BLOCK_SIZE threads
//
// Each thread owns one output element:
//   out[h] = sum_k( bf16(rw_fp32[k]) * down_scratch[k * stride + h] )
// Accumulate in FP32, write BF16. Route weight cast to BF16 first (match Python).
// ─────────────────────────────────────────────────────────────────────────────

__global__ void weighted_reduce_bf16_kernel(
    const __nv_bfloat16* __restrict__ down_data,  // [8, 2048] contiguous (squeezed)
    const float* __restrict__ routing_weights,    // [8] FP32
    __nv_bfloat16* __restrict__ out,              // [2048] BF16
    int64_t stride_k                              // stride between experts (= 2048 for contiguous)
) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= kHidden) return;

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        // Cast route weight to BF16 then back to FP32 (matches Python rw[k].to(bf16))
        const __nv_bfloat16 rw_bf16 = __float2bfloat16(routing_weights[k]);
        const float rw = __bfloat162float(rw_bf16);
        const float val = __bfloat162float(down_data[k * stride_k + h]);
        acc += rw * val;
    }
    out[h] = __float2bfloat16(acc);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Python entry point: TRUE allocation-free hot path
// ─────────────────────────────────────────────────────────────────────────────

torch::Tensor lynn_native_moe_packed_pretransposed_graphsafe_v3(
    torch::Tensor x,                   // [1, 2048] BF16
    torch::Tensor routing_weights,     // [8] float32
    torch::Tensor W_fused_T,           // [2048, 8192] BF16 contiguous
    torch::Tensor W_down_T,            // [8, 512, 2048] BF16 contiguous
    torch::Tensor gate_up_scratch,     // [1, 8192] BF16
    torch::Tensor inter_scratch,       // [8, 1, 512] BF16
    torch::Tensor down_scratch,        // [8, 1, 2048] BF16
    torch::Tensor out                  // [2048] BF16
) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1 && x.size(1) == kHidden);
    TORCH_CHECK(routing_weights.is_cuda() && routing_weights.scalar_type() == torch::kFloat32);
    TORCH_CHECK(W_fused_T.is_cuda() && W_fused_T.is_contiguous());
    TORCH_CHECK(W_fused_T.size(0) == kHidden && W_fused_T.size(1) == kGateUpFused);
    TORCH_CHECK(W_down_T.is_cuda() && W_down_T.is_contiguous());
    TORCH_CHECK(W_down_T.size(0) == kTopK && W_down_T.size(1) == kIntermediate && W_down_T.size(2) == kHidden);
    TORCH_CHECK(gate_up_scratch.is_cuda() && gate_up_scratch.is_contiguous());
    TORCH_CHECK(gate_up_scratch.size(0) == 1 && gate_up_scratch.size(1) == kGateUpFused);
    TORCH_CHECK(inter_scratch.is_cuda() && inter_scratch.is_contiguous());
    TORCH_CHECK(inter_scratch.size(0) == kTopK && inter_scratch.size(1) == 1 && inter_scratch.size(2) == kIntermediate);
    TORCH_CHECK(down_scratch.is_cuda() && down_scratch.is_contiguous());
    TORCH_CHECK(down_scratch.size(0) == kTopK && down_scratch.size(1) == 1 && down_scratch.size(2) == kHidden);
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == kHidden);

    // Stage 1: gate_up GEMM — writes into gate_up_scratch, no allocation
    torch::mm_out(gate_up_scratch, x, W_fused_T);

    // View + slice (metadata only, no allocation)
    auto gate_up_3d = gate_up_scratch.view({kTopK, 1, 1024});
    auto gate_view = gate_up_3d.slice(2, 0, kIntermediate);
    auto up_view = gate_up_3d.slice(2, kIntermediate, 1024);

    // SiLU + multiply — writes into inter_scratch, no allocation
    at::silu_out(inter_scratch, gate_view);
    inter_scratch.mul_(up_view);

    // Stage 2: down GEMM — writes into down_scratch, no allocation
    torch::bmm_out(down_scratch, inter_scratch, W_down_T);

    // Stage 3: custom reduce kernel — reads down_scratch + routing_weights, writes out
    // down_scratch is [8, 1, 2048] contiguous → stride between experts = 1*2048 = 2048
    constexpr int BLOCK = 256;
    dim3 grid((kHidden + BLOCK - 1) / BLOCK);
    weighted_reduce_bf16_kernel<<<grid, BLOCK>>>(
        reinterpret_cast<const __nv_bfloat16*>(down_scratch.data_ptr<at::BFloat16>()),
        routing_weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        down_scratch.stride(0)  // stride between expert slots in elements
    );

    return out;
}
