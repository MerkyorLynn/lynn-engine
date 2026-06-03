#include <torch/extension.h>

namespace {

constexpr int64_t kHidden = 2048;
constexpr int64_t kIntermediate = 512;
constexpr int64_t kGateUpRows = kIntermediate * 2;

void check_cuda_tensor(const torch::Tensor& t, const char* name, c10::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has wrong dtype");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous for the P4 hot-path ABI");
}

}  // namespace

void lynn_native_active_moe_fused_zero_shadow_out_contract(
    torch::Tensor hidden,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale,
    torch::Tensor out,
    int64_t tile_tokens,
    int64_t tile_inter,
    int64_t tile_hidden) {
  check_cuda_tensor(hidden, "hidden", torch::kBFloat16);
  check_cuda_tensor(expert_ids, "expert_ids", torch::kInt32);
  check_cuda_tensor(routing_weights, "routing_weights", torch::kFloat32);
  check_cuda_tensor(gate_up_packed, "gate_up_packed", torch::kUInt8);
  check_cuda_tensor(gate_up_scale, "gate_up_scale", torch::kFloat32);
  check_cuda_tensor(gate_up_global_scale, "gate_up_global_scale", torch::kFloat32);
  check_cuda_tensor(down_packed, "down_packed", torch::kUInt8);
  check_cuda_tensor(down_scale, "down_scale", torch::kFloat32);
  check_cuda_tensor(down_global_scale, "down_global_scale", torch::kFloat32);
  check_cuda_tensor(out, "out", torch::kBFloat16);

  TORCH_CHECK(hidden.dim() == 2, "hidden must be [T, 2048]");
  const int64_t tokens = hidden.size(0);
  TORCH_CHECK(tokens > 0, "hidden token dimension must be non-empty");
  TORCH_CHECK(hidden.size(1) == kHidden, "hidden dim must be 2048");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == tokens && out.size(1) == kHidden, "out must be [T, 2048]");

  TORCH_CHECK(expert_ids.dim() == 2, "expert_ids must be [T, top_k]");
  TORCH_CHECK(routing_weights.dim() == 2, "routing_weights must be [T, top_k]");
  TORCH_CHECK(expert_ids.size(0) == tokens, "expert_ids token dimension must match hidden");
  TORCH_CHECK(routing_weights.size(0) == tokens, "routing_weights token dimension must match hidden");
  TORCH_CHECK(routing_weights.size(1) == expert_ids.size(1), "routing_weights top_k must match expert_ids");
  TORCH_CHECK(expert_ids.size(1) > 0, "top_k must be non-empty");

  TORCH_CHECK(gate_up_packed.dim() == 3, "gate_up_packed must be [E, 1024, 1024]");
  TORCH_CHECK(gate_up_scale.dim() == 3, "gate_up_scale must be [E, 1024, 128]");
  TORCH_CHECK(down_packed.dim() == 3, "down_packed must be [E, 2048, 256]");
  TORCH_CHECK(down_scale.dim() == 3, "down_scale must be [E, 2048, 32]");
  const int64_t experts = gate_up_packed.size(0);
  TORCH_CHECK(experts > 0, "expert dimension must be non-empty");
  TORCH_CHECK(gate_up_scale.size(0) == experts, "gate_up_scale expert dimension must match gate_up_packed");
  TORCH_CHECK(down_packed.size(0) == experts, "down_packed expert dimension must match gate_up_packed");
  TORCH_CHECK(down_scale.size(0) == experts, "down_scale expert dimension must match gate_up_packed");

  TORCH_CHECK(gate_up_packed.size(1) == kGateUpRows, "gate_up_packed row dim must be 1024");
  TORCH_CHECK(gate_up_packed.size(2) == kHidden / 2, "gate_up_packed packed hidden dim must be 1024");
  TORCH_CHECK(gate_up_scale.size(1) == kGateUpRows, "gate_up_scale row dim must be 1024");
  TORCH_CHECK(gate_up_scale.size(2) == kHidden / 16, "gate_up_scale group dim must be 128");
  TORCH_CHECK(down_packed.size(1) == kHidden, "down_packed row dim must be 2048");
  TORCH_CHECK(down_packed.size(2) == kIntermediate / 2, "down_packed packed inter dim must be 256");
  TORCH_CHECK(down_scale.size(1) == kHidden, "down_scale row dim must be 2048");
  TORCH_CHECK(down_scale.size(2) == kIntermediate / 16, "down_scale group dim must be 32");
  TORCH_CHECK(gate_up_global_scale.numel() == 1, "gate_up_global_scale must be scalar");
  TORCH_CHECK(down_global_scale.numel() == 1, "down_global_scale must be scalar");

  TORCH_CHECK(tile_tokens > 0, "tile_tokens must be positive");
  TORCH_CHECK(
      tile_inter == 1 || tile_inter == 2 || tile_inter == 4 || tile_inter == 8 || tile_inter == 16,
      "tile_inter must be one of {1, 2, 4, 8, 16}");
  TORCH_CHECK(
      tile_hidden == 1 || tile_hidden == 2 || tile_hidden == 4 || tile_hidden == 8 || tile_hidden == 16,
      "tile_hidden must be one of {1, 2, 4, 8, 16}");

  TORCH_CHECK(
      false,
      "active_moe_fused_zero_shadow_out_contract passed all packed-NVFP4 shape/layout checks, "
      "but the P4 fused 4-bit zero-shadow CUDA kernel is not implemented yet. "
      "This is the caller-owned-output C++ hot-path ABI: replace only the inner math, "
      "do not add BF16 expert shadows or Python/Triton fallback inside this symbol.");
}
