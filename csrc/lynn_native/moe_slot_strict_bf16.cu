/**
 * Lynn Engine · Native MoE Slot Output-Owned Strict BF16
 *
 * Strategy: Use cuBLAS (via torch::mm) for the matmuls to exactly match
 * PyTorch F.linear accumulation semantics, while keeping the output-owned
 * non-atomic dispatch structure.
 *
 * This eliminates:
 * 1. Thread-striped accumulation ordering mismatch
 * 2. BF16 intermediate round-trip (keep FP32 inter in registers)
 *
 * The "kernel" is really a cuBLAS-orchestrated sequence:
 *   For each slot k in [0..8):
 *     gate_up = x @ slot_gate_up_weight[k].T   (cuBLAS BF16 GEMM)
 *     gate, up = split(gate_up)
 *     inter = silu(gate) * up
 *     down_out = inter @ slot_down_weight[k].T  (cuBLAS BF16 GEMM)
 *     out += routing_weights[k] * down_out
 *
 * This matches F.linear(x, w) exactly because that's what F.linear calls.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

namespace {
constexpr int kHidden = 2048;
constexpr int kIntermediate = 512;
constexpr int kGateUpRows = kIntermediate * 2;  // 1024
}

torch::Tensor lynn_native_moe_slot_strict_bf16(
    torch::Tensor x,                   // [1, 2048] or [2048] BF16
    torch::Tensor routing_weights,     // [top_k] float32
    torch::Tensor slot_gate_up,        // [top_k, 1024, 2048] BF16
    torch::Tensor slot_down            // [top_k, 2048, 512] BF16
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(routing_weights.is_cuda() && routing_weights.scalar_type() == torch::kFloat32);
    TORCH_CHECK(slot_gate_up.is_cuda() && slot_gate_up.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(slot_down.is_cuda() && slot_down.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(slot_gate_up.dim() == 3);
    TORCH_CHECK(slot_down.dim() == 3);

    const int top_k = slot_gate_up.size(0);
    TORCH_CHECK(routing_weights.size(0) == top_k);
    TORCH_CHECK(slot_down.size(0) == top_k);
    TORCH_CHECK(slot_gate_up.size(1) == kGateUpRows);
    TORCH_CHECK(slot_gate_up.size(2) == kHidden);
    TORCH_CHECK(slot_down.size(1) == kHidden);
    TORCH_CHECK(slot_down.size(2) == kIntermediate);

    // Reshape x to [1, hidden] if needed
    auto x_2d = x.dim() == 1 ? x.unsqueeze(0) : x;  // [1, 2048]
    TORCH_CHECK(x_2d.size(0) == 1 && x_2d.size(1) == kHidden);

    // Output accumulator in BF16 (matching F.linear output dtype)
    auto out = torch::zeros({1, kHidden},
                            torch::TensorOptions().device(x.device()).dtype(torch::kBFloat16));

    // Process each slot sequentially (output-owned: no parallelism needed across slots
    // for single-token decode — the parallelism is within each cuBLAS call)
    for (int k = 0; k < top_k; ++k) {
        float route_w = routing_weights[k].item<float>();

        // gate_up = x @ slot_gate_up[k].T
        // F.linear(x, w) = x @ w.T, so: mm(x_2d, w.T) = mm([1,2048], [2048,1024]) = [1,1024]
        auto w_gate_up_k = slot_gate_up[k];  // [1024, 2048]
        auto gate_up = torch::mm(x_2d, w_gate_up_k.t());  // [1, 1024]

        // Split gate and up
        auto chunks = gate_up.chunk(2, /*dim=*/1);  // each [1, 512]
        auto gate = chunks[0];
        auto up = chunks[1];

        // SiLU(gate) * up — element-wise, stays in whatever dtype cuBLAS returned
        auto inter = torch::silu(gate) * up;  // [1, 512]

        // down_out = inter @ slot_down[k].T
        // slot_down[k] is [2048, 512] (F.linear weight layout: [out_features, in_features])
        auto w_down_k = slot_down[k];  // [2048, 512]
        auto down_out = torch::mm(inter, w_down_k.t());  // [1, 2048]

        // Weighted accumulate — match Python: out += down_out * bf16(route_w)
        // Using .add_(tensor, alpha) would apply float32 alpha which differs from
        // Python's `ffn * rw[k].to(h.dtype)` which first truncates to BF16.
        auto route_bf16 = torch::tensor({route_w},
            torch::TensorOptions().device(x.device()).dtype(torch::kBFloat16));
        out.add_(down_out * route_bf16);
    }

    // Return [2048] if input was [2048], else [1, 2048]
    if (x.dim() == 1) {
        return out.squeeze(0);
    }
    return out;
}
