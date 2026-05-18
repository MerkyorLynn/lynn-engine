/**
 * Lynn Engine · MoE Slot TensorCore Probe (P139)
 *
 * Optimal dispatch for M=1 decode:
 * - Stage 1 (gate_up): Fuse 8 slots into single large GEMM
 *   mm(x[1,2048], W_fused[2048, 8192]) → [1, 8192] → reshape [8, 1024]
 *   One cuBLAS launch, TensorCore-friendly (N=8192 >> M=1)
 *
 * - Stage 2 (down): batched bmm
 *   bmm(inter[8,1,512], slot_down_T[8,512,2048]) → [8,1,2048]
 *   One cuBLAS launch
 *
 * - Reduce: out = sum_k(bf16(rw_k) * down_out_k)
 *
 * Total: 2 cuBLAS launches. Expected: ~0.05ms on Blackwell.
 * Routing weight as BF16 (matches slot-order PyTorch semantics).
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

namespace {
constexpr int kHidden = 2048;
constexpr int kIntermediate = 512;
constexpr int kGateUpRows = kIntermediate * 2;
}

torch::Tensor lynn_native_moe_slot_tensorcore_probe(
    torch::Tensor x,                   // [2048] or [1, 2048] BF16
    torch::Tensor routing_weights,     // [top_k] float32
    torch::Tensor slot_gate_up,        // [top_k, 1024, 2048] BF16
    torch::Tensor slot_down            // [top_k, 2048, 512] BF16
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(routing_weights.is_cuda() && routing_weights.scalar_type() == torch::kFloat32);
    TORCH_CHECK(slot_gate_up.is_cuda() && slot_gate_up.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(slot_down.is_cuda() && slot_down.scalar_type() == torch::kBFloat16);

    const int top_k = slot_gate_up.size(0);
    TORCH_CHECK(routing_weights.size(0) == top_k);
    TORCH_CHECK(slot_down.size(0) == top_k);
    TORCH_CHECK(slot_gate_up.size(1) == kGateUpRows);
    TORCH_CHECK(slot_gate_up.size(2) == kHidden);
    TORCH_CHECK(slot_down.size(1) == kHidden);
    TORCH_CHECK(slot_down.size(2) == kIntermediate);

    auto x_2d = x.dim() == 1 ? x.unsqueeze(0) : x;  // [1, 2048]

    // ── Stage 1: Fused gate_up GEMM ──
    // Reshape slot_gate_up[top_k, 1024, 2048] → [top_k*1024, 2048]
    // mm(x[1,2048], W_fused.T[2048, top_k*1024]) → [1, top_k*1024]
    auto W_fused = slot_gate_up.reshape({top_k * kGateUpRows, kHidden});  // [8192, 2048]
    auto gate_up_flat = torch::mm(x_2d, W_fused.t());  // [1, 8192] — ONE cuBLAS launch

    // Reshape to [top_k, 1, 1024] then split gate/up
    auto gate_up_3d = gate_up_flat.view({top_k, 1, kGateUpRows});
    auto chunks = gate_up_3d.chunk(2, /*dim=*/2);  // gate [8,1,512], up [8,1,512]
    auto inter = torch::silu(chunks[0]) * chunks[1];  // [top_k, 1, 512]

    // ── Stage 2: Batched down GEMM ──
    // slot_down transposed: [top_k, 512, 2048]
    auto down_t = slot_down.transpose(1, 2).contiguous();
    // bmm: [top_k, 1, 512] x [top_k, 512, 2048] → [top_k, 1, 2048]
    auto down_out = torch::bmm(inter, down_t);  // ONE cuBLAS launch

    // ── Weighted reduce (bf16 routing weight, matches slot-order PyTorch) ──
    auto rw_bf16 = routing_weights.to(torch::kBFloat16).view({top_k, 1, 1});
    auto weighted = down_out * rw_bf16;  // [top_k, 1, 2048]
    auto out = weighted.sum(/*dim=*/0);   // [1, 2048]

    if (x.dim() == 1) {
        return out.squeeze(0);
    }
    return out;
}
