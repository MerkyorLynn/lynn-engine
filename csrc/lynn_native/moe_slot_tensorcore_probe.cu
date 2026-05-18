/**
 * Lynn Engine · MoE Slot TensorCore Probe (P139)
 *
 * Replace 16 sequential cuBLAS calls with 2 batched GEMMs (torch::bmm)
 * so cuBLAS dispatches all 8 expert slots in ONE TensorCore kernel launch.
 *
 * Stage 1: gate_up = bmm(x_exp[8,1,2048], slot_gate_up_T[8,2048,1024]) → [8,1,1024]
 * Stage 2: down_out = bmm(inter[8,1,512], slot_down_T[8,512,2048]) → [8,1,2048]
 * Reduce: out = sum_k(routing_w[k] * down_out[k])
 *
 * 2 cuBLAS launches total. Blackwell TensorCores handle the BF16 matmuls.
 * Routing weight applied as BF16 multiply (matches slot-order PyTorch semantics).
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

    // ── Stage 1: Batched gate_up GEMM ──
    // x_expanded: [top_k, 1, 2048]
    auto x_exp = x_2d.unsqueeze(0).expand({top_k, 1, kHidden}).contiguous();
    // slot_gate_up transposed: [top_k, 2048, 1024]
    auto gate_up_t = slot_gate_up.transpose(1, 2).contiguous();
    // bmm: [top_k, 1, 2048] x [top_k, 2048, 1024] → [top_k, 1, 1024]
    auto gate_up_all = torch::bmm(x_exp, gate_up_t);

    // Split + SiLU: gate [top_k, 1, 512], up [top_k, 1, 512]
    auto chunks = gate_up_all.chunk(2, /*dim=*/2);
    auto inter = torch::silu(chunks[0]) * chunks[1];  // [top_k, 1, 512]

    // ── Stage 2: Batched down GEMM ──
    // slot_down transposed: [top_k, 512, 2048]
    auto down_t = slot_down.transpose(1, 2).contiguous();
    // bmm: [top_k, 1, 512] x [top_k, 512, 2048] → [top_k, 1, 2048]
    auto down_out = torch::bmm(inter, down_t);

    // ── Weighted reduce (match slot-order PyTorch: rw as BF16) ──
    auto rw_bf16 = routing_weights.to(torch::kBFloat16).view({top_k, 1, 1});
    auto weighted = down_out * rw_bf16;   // [top_k, 1, 2048]
    auto out = weighted.sum(/*dim=*/0);    // [1, 2048]

    if (x.dim() == 1) {
        return out.squeeze(0);
    }
    return out;
}
