/**
 * Lynn Engine · P142 Graph-Safe Pretransposed MoE V3
 *
 * All scratch/output tensors are caller-owned. NO allocation in hot path.
 * Safe for CUDA graph capture.
 *
 * Caller provides (preallocated once at model load):
 *   W_fused_T:       [2048, 8192] BF16 contiguous (dequant+pretranspose at load)
 *   W_down_T:        [8, 512, 2048] BF16 contiguous
 *   gate_up_scratch: [1, 8192] BF16
 *   inter_scratch:   [8, 1, 512] BF16
 *   down_scratch:    [8, 1, 2048] BF16
 *   out:             [2048] BF16
 *
 * Hot path: mm_out + view + silu_out*= + bmm_out + weighted reduce
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

namespace {
constexpr int kHidden = 2048;
constexpr int kIntermediate = 512;
constexpr int kGateUpFused = 8192;
constexpr int kTopK = 8;
}

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
    TORCH_CHECK(W_fused_T.is_cuda() && W_fused_T.is_contiguous());
    TORCH_CHECK(W_fused_T.size(0) == kHidden && W_fused_T.size(1) == kGateUpFused);
    TORCH_CHECK(W_down_T.is_cuda() && W_down_T.is_contiguous());
    TORCH_CHECK(W_down_T.size(0) == kTopK && W_down_T.size(1) == kIntermediate && W_down_T.size(2) == kHidden);
    TORCH_CHECK(gate_up_scratch.size(0) == 1 && gate_up_scratch.size(1) == kGateUpFused);
    TORCH_CHECK(inter_scratch.size(0) == kTopK && inter_scratch.size(1) == 1 && inter_scratch.size(2) == kIntermediate);
    TORCH_CHECK(down_scratch.size(0) == kTopK && down_scratch.size(1) == 1 && down_scratch.size(2) == kHidden);
    TORCH_CHECK(out.size(0) == kHidden);

    // Stage 1: mm_out — no allocation
    torch::mm_out(gate_up_scratch, x, W_fused_T);

    // View as [8, 1, 1024] then split gate/up
    auto gate_up_3d = gate_up_scratch.view({kTopK, 1, 1024});
    auto gate_view = gate_up_3d.slice(2, 0, kIntermediate);
    auto up_view = gate_up_3d.slice(2, kIntermediate, 1024);

    // inter_scratch = silu(gate) * up — in-place ops
    at::silu_out(inter_scratch, gate_view);
    inter_scratch.mul_(up_view);

    // Stage 2: bmm_out — no allocation
    torch::bmm_out(down_scratch, inter_scratch, W_down_T);

    // Weighted reduce: vectorized — single sum op instead of 8 add_ calls
    // down_scratch[8,1,2048] → squeeze → [8,2048]
    // rw_bf16[8,1] broadcast → weighted[8,2048] → sum(dim=0) → [2048]
    auto down_2d = down_scratch.squeeze(1);  // [8, 2048] view
    auto rw_bf16 = routing_weights.to(torch::kBFloat16).view({kTopK, 1});
    auto weighted = down_2d * rw_bf16;
    auto summed = weighted.sum(0);  // [2048]
    out.copy_(summed);

    return out;
}
