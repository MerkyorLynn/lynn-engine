#include <torch/extension.h>

torch::Tensor lynn_native_add_one(torch::Tensor x);
torch::Tensor lynn_native_gate_up_silu_scalar(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale);
torch::Tensor lynn_native_down_weighted_sum_scalar(
    torch::Tensor inter,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale);
torch::Tensor lynn_native_down_weighted_sum_tile_scalar(
    torch::Tensor inter,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale,
    int64_t tile_hidden);
torch::Tensor lynn_native_active_moe_scalar_contract(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale);
torch::Tensor lynn_native_active_moe_fused_atomic_scalar(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_one", &lynn_native_add_one, "Lynn native CUDA extension smoke kernel");
  m.def(
      "gate_up_silu_scalar",
      &lynn_native_gate_up_silu_scalar,
      "Reference CUDA scalar gate/up kernel for packed NVFP4 active experts");
  m.def(
      "down_weighted_sum_scalar",
      &lynn_native_down_weighted_sum_scalar,
      "Reference CUDA scalar down weighted-sum kernel for packed NVFP4 active experts");
  m.def(
      "down_weighted_sum_tile_scalar",
      &lynn_native_down_weighted_sum_tile_scalar,
      "P48 tile-hidden non-atomic CUDA scalar down weighted-sum probe for packed NVFP4 active experts");
  m.def(
      "active_moe_scalar_contract",
      &lynn_native_active_moe_scalar_contract,
      "Reference one-call active MoE contract for future grouped native FP4 kernels");
  m.def(
      "active_moe_fused_atomic_scalar",
      &lynn_native_active_moe_fused_atomic_scalar,
      "P46 fused atomic scalar active MoE probe for packed NVFP4 experts");
}
