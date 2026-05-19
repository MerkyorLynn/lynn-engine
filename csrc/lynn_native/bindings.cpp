#include <torch/extension.h>

torch::Tensor lynn_native_add_one(torch::Tensor x);
torch::Tensor lynn_native_gate_up_silu_scalar(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale);
torch::Tensor lynn_native_gate_up_silu_tile_inter_scalar(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    int64_t tile_inter);
torch::Tensor lynn_native_gate_up_silu_tile_inter_threads_scalar(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    int64_t tile_inter,
    int64_t threads);
torch::Tensor lynn_native_gate_up_silu_split16_topk_fp4(
    torch::Tensor act_packed,
    torch::Tensor act_scale,
    torch::Tensor expert_ids,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    int64_t scale_byte);
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
torch::Tensor lynn_native_down_grouped_per16_reference(
    torch::Tensor inter,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale);
torch::Tensor lynn_native_down_grouped_per16_tile_reference(
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
torch::Tensor lynn_native_active_moe_grouped_per16_contract(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale);
torch::Tensor lynn_native_active_moe_grouped_per16_tile_reference(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale,
    int64_t tile_inter,
    int64_t tile_hidden);
torch::Tensor lynn_native_active_moe_grouped_per16_fused_contract(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale);
torch::Tensor lynn_native_active_moe_grouped_per16_nonatomic_reference(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale,
    int64_t tile_inter,
    int64_t tile_hidden);
torch::Tensor lynn_native_active_moe_grouped_per16_nonatomic_out_reference(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale,
    torch::Tensor inter_out,
    torch::Tensor out,
    int64_t tile_inter,
    int64_t tile_hidden);
torch::Tensor lynn_native_moe_output_owned_bf16(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor gate_up_weight,
    torch::Tensor down_weight);
torch::Tensor lynn_native_moe_slot_output_owned_bf16(
    torch::Tensor x,
    torch::Tensor routing_weights,
    torch::Tensor slot_gate_up,
    torch::Tensor slot_down);
torch::Tensor lynn_native_moe_slot_gate_up_inter_bf16(
    torch::Tensor x,
    torch::Tensor slot_gate_up);
torch::Tensor lynn_native_moe_slot_down_weighted_sum_bf16(
    torch::Tensor inter,
    torch::Tensor routing_weights,
    torch::Tensor slot_down);
torch::Tensor lynn_native_moe_slot_strict_bf16(
    torch::Tensor x,
    torch::Tensor routing_weights,
    torch::Tensor slot_gate_up,
    torch::Tensor slot_down);

// P139 TensorCore batched-bmm slot MoE probe
torch::Tensor lynn_native_moe_slot_tensorcore_probe(
    torch::Tensor x, torch::Tensor routing_weights,
    torch::Tensor slot_gate_up, torch::Tensor slot_down);
// P139b pretransposed variant (zero-overhead hot path)
torch::Tensor lynn_native_moe_slot_tensorcore_pretransposed(
    torch::Tensor x, torch::Tensor routing_weights,
    torch::Tensor W_fused_T, torch::Tensor W_down_T);

// P140 packed NVFP4 slot MoE probe
torch::Tensor lynn_native_moe_slot_packed_nvfp4_probe(
    torch::Tensor x, torch::Tensor routing_weights,
    torch::Tensor slot_gate_up_packed, torch::Tensor slot_gate_up_scale,
    torch::Tensor slot_gate_up_global_scale,
    torch::Tensor slot_down_packed, torch::Tensor slot_down_scale,
    torch::Tensor slot_down_global_scale);
torch::Tensor lynn_native_moe_slot_packed_nvfp4_inter_probe(
    torch::Tensor x,
    torch::Tensor slot_gate_up_packed,
    torch::Tensor slot_gate_up_scale,
    torch::Tensor slot_gate_up_global_scale);
torch::Tensor lynn_native_moe_slot_packed_nvfp4_inter_triton_order_probe(
    torch::Tensor x,
    torch::Tensor slot_gate_up_packed,
    torch::Tensor slot_gate_up_scale,
    torch::Tensor slot_gate_up_global_scale);
torch::Tensor lynn_native_moe_slot_packed_nvfp4_raw_accum_probe(
    torch::Tensor x,
    torch::Tensor slot_gate_up_packed,
    torch::Tensor slot_gate_up_scale,
    torch::Tensor slot_gate_up_global_scale);
torch::Tensor lynn_native_moe_slot_packed_nvfp4_raw_accum_triton_order_probe(
    torch::Tensor x,
    torch::Tensor slot_gate_up_packed,
    torch::Tensor slot_gate_up_scale,
    torch::Tensor slot_gate_up_global_scale);
torch::Tensor lynn_native_moe_slot_packed_nvfp4_partial_accum_triton_order_probe(
    torch::Tensor x,
    torch::Tensor slot_gate_up_packed,
    torch::Tensor slot_gate_up_scale,
    torch::Tensor slot_gate_up_global_scale);
torch::Tensor lynn_native_moe_slot_packed_nvfp4_term_trace_probe(
    torch::Tensor x,
    torch::Tensor slot_gate_up_packed,
    torch::Tensor slot_gate_up_scale,
    torch::Tensor slot_gate_up_global_scale,
    int64_t slot,
    int64_t row,
    int64_t kind,
    int64_t hidden_block);
torch::Tensor lynn_native_moe_slot_packed_nvfp4_down_probe(
    torch::Tensor inter,
    torch::Tensor routing_weights,
    torch::Tensor slot_down_packed,
    torch::Tensor slot_down_scale,
    torch::Tensor slot_down_global_scale);
// P142 graph-safe pretransposed MoE V3
torch::Tensor lynn_native_moe_packed_pretransposed_graphsafe_v3(
    torch::Tensor x, torch::Tensor routing_weights,
    torch::Tensor W_fused_T, torch::Tensor W_down_T,
    torch::Tensor gate_up_scratch, torch::Tensor inter_scratch,
    torch::Tensor down_scratch, torch::Tensor out);

// P145 graph-safe scalar MoE V3.2 (caller-owned scratch, exact reference)
void lynn_native_gate_up_silu_scalar_out(
    torch::Tensor x, torch::Tensor expert_ids,
    torch::Tensor gate_up_packed, torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale, torch::Tensor out);
void lynn_native_down_weighted_sum_scalar_out(
    torch::Tensor inter, torch::Tensor expert_ids,
    torch::Tensor routing_weights, torch::Tensor down_packed,
    torch::Tensor down_scale, torch::Tensor down_global_scale,
    torch::Tensor out);
void lynn_native_active_moe_scalar_out_reference(
    torch::Tensor x, torch::Tensor expert_ids,
    torch::Tensor routing_weights, torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale, torch::Tensor gate_up_global_scale,
    torch::Tensor down_packed, torch::Tensor down_scale,
    torch::Tensor down_global_scale, torch::Tensor inter_out,
    torch::Tensor out);

// P191 Dense FP4xFP8 PoC
torch::Tensor lynn_dense_fp4xfp8_scalar_reference(
    torch::Tensor act_fp8, torch::Tensor act_scale,
    torch::Tensor weight_packed, torch::Tensor weight_scale,
    torch::Tensor weight_global);
torch::Tensor lynn_dense_fp4xfp8_mma_probe(
    torch::Tensor act_fp8, torch::Tensor weight_packed,
    int64_t M, int64_t N, int64_t K);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_one", &lynn_native_add_one, "Lynn native CUDA extension smoke kernel");
  m.def(
      "gate_up_silu_scalar",
      &lynn_native_gate_up_silu_scalar,
      "Reference CUDA scalar gate/up kernel for packed NVFP4 active experts");
  m.def(
      "gate_up_silu_tile_inter_scalar",
      &lynn_native_gate_up_silu_tile_inter_scalar,
      "P55 tile-inter CUDA scalar gate/up probe for packed NVFP4 active experts");
  m.def(
      "gate_up_silu_tile_inter_threads_scalar",
      &lynn_native_gate_up_silu_tile_inter_threads_scalar,
      "P75 tile-inter/thread sweep CUDA scalar gate/up probe for packed NVFP4 active experts");
  m.def(
      "gate_up_silu_split16_topk_fp4",
      &lynn_native_gate_up_silu_split16_topk_fp4,
      "P98 opt-in SM120a split16 FP4 MMA gate/up backend for packed NVFP4 active experts");
  m.def(
      "down_weighted_sum_scalar",
      &lynn_native_down_weighted_sum_scalar,
      "Reference CUDA scalar down weighted-sum kernel for packed NVFP4 active experts");
  m.def(
      "down_weighted_sum_tile_scalar",
      &lynn_native_down_weighted_sum_tile_scalar,
      "P48 tile-hidden non-atomic CUDA scalar down weighted-sum probe for packed NVFP4 active experts");
  m.def(
      "down_grouped_per16_reference",
      &lynn_native_down_grouped_per16_reference,
      "P66 reference ABI for future grouped per-16 native-FP4 down projection");
  m.def(
      "down_grouped_per16_tile_reference",
      &lynn_native_down_grouped_per16_tile_reference,
      "P67 tile reference ABI for future grouped per-16 native-FP4 down projection");
  m.def(
      "active_moe_scalar_contract",
      &lynn_native_active_moe_scalar_contract,
      "Reference one-call active MoE contract for future grouped native FP4 kernels");
  m.def(
      "active_moe_fused_atomic_scalar",
      &lynn_native_active_moe_fused_atomic_scalar,
      "P46 fused atomic scalar active MoE probe for packed NVFP4 experts");
  m.def(
      "active_moe_grouped_per16_contract",
      &lynn_native_active_moe_grouped_per16_contract,
      "P65 guarded ABI for the future grouped per-16 native-FP4 active expert FFN");
  m.def(
      "active_moe_grouped_per16_tile_reference",
      &lynn_native_active_moe_grouped_per16_tile_reference,
      "P68 tiled reference ABI for future grouped per-16 native-FP4 active expert FFN");
  m.def(
      "active_moe_grouped_per16_fused_contract",
      &lynn_native_active_moe_grouped_per16_fused_contract,
      "P70 fail-loud ABI for the true fused grouped per-16 native-FP4 active expert FFN");
  m.def(
      "active_moe_grouped_per16_nonatomic_reference",
      &lynn_native_active_moe_grouped_per16_nonatomic_reference,
      "P73 non-atomic native-owned scratch reference for grouped per-16 active expert FFN");
  m.def(
      "active_moe_grouped_per16_nonatomic_out_reference",
      &lynn_native_active_moe_grouped_per16_nonatomic_out_reference,
      "P136 graph-safe caller-owned scratch reference for grouped per-16 active expert FFN");
  m.def(
      "moe_output_owned_bf16",
      &lynn_native_moe_output_owned_bf16,
      "P134 candidate output-owned BF16 routed active-MoE kernel");
  m.def(
      "moe_slot_output_owned_bf16",
      &lynn_native_moe_slot_output_owned_bf16,
      "P135/P136 slot-repacked output-owned BF16 routed active-MoE kernel");
  m.def(
      "moe_slot_gate_up_inter_bf16",
      &lynn_native_moe_slot_gate_up_inter_bf16,
      "P137 diagnostic slot-repacked BF16 gate/up intermediate kernel");
  m.def(
      "moe_slot_down_weighted_sum_bf16",
      &lynn_native_moe_slot_down_weighted_sum_bf16,
      "P137 diagnostic slot-repacked BF16 down weighted-sum kernel");
  m.def(
      "moe_slot_strict_bf16",
      &lynn_native_moe_slot_strict_bf16,
      "P138 cuBLAS-matched strict BF16 slot MoE reference");
  m.def(
      "moe_slot_tensorcore_probe",
      &lynn_native_moe_slot_tensorcore_probe,
      "P139 TensorCore batched-bmm slot MoE probe (2 launches)");
  m.def(
      "moe_slot_tensorcore_pretransposed",
      &lynn_native_moe_slot_tensorcore_pretransposed,
      "P139b pretransposed slot MoE (zero-overhead hot path, 2 launches)");
  m.def(
      "moe_slot_packed_nvfp4_probe",
      &lynn_native_moe_slot_packed_nvfp4_probe,
      "P140 packed NVFP4 slot MoE probe (correctness-first, scalar dequant)");
  m.def(
      "moe_slot_packed_nvfp4_inter_probe",
      &lynn_native_moe_slot_packed_nvfp4_inter_probe,
      "P153 packed NVFP4 slot MoE gate/up intermediate probe");
  m.def(
      "moe_slot_packed_nvfp4_inter_triton_order_probe",
      &lynn_native_moe_slot_packed_nvfp4_inter_triton_order_probe,
      "P154 packed NVFP4 slot MoE gate/up intermediate probe with Triton-like hidden-block reduction");
  m.def(
      "moe_slot_packed_nvfp4_raw_accum_probe",
      &lynn_native_moe_slot_packed_nvfp4_raw_accum_probe,
      "P155 packed NVFP4 slot MoE raw FP32 gate/up accumulator probe");
  m.def(
      "moe_slot_packed_nvfp4_raw_accum_triton_order_probe",
      &lynn_native_moe_slot_packed_nvfp4_raw_accum_triton_order_probe,
      "P155 packed NVFP4 slot MoE raw FP32 gate/up accumulator probe with Triton-like hidden-block reduction");
  m.def(
      "moe_slot_packed_nvfp4_partial_accum_triton_order_probe",
      &lynn_native_moe_slot_packed_nvfp4_partial_accum_triton_order_probe,
      "P160 packed NVFP4 slot MoE per-hidden-block FP32 partial accumulator trace with Triton-like reduction");
  m.def(
      "moe_slot_packed_nvfp4_term_trace_probe",
      &lynn_native_moe_slot_packed_nvfp4_term_trace_probe,
      "P161 packed NVFP4 slot MoE per-term FP32 trace for a single gate/up row");
  m.def(
      "moe_slot_packed_nvfp4_down_probe",
      &lynn_native_moe_slot_packed_nvfp4_down_probe,
      "P153 packed NVFP4 slot MoE down weighted-sum probe");
  m.def(
      "moe_packed_pretransposed_graphsafe_v3",
      &lynn_native_moe_packed_pretransposed_graphsafe_v3,
      "P142 graph-safe pretransposed MoE V3 (caller-owned scratch, no alloc)");
  m.def(
      "gate_up_silu_scalar_out",
      &lynn_native_gate_up_silu_scalar_out,
      "P145 graph-safe scalar gate/up (caller-owned out, no alloc)");
  m.def(
      "down_weighted_sum_scalar_out",
      &lynn_native_down_weighted_sum_scalar_out,
      "P145 graph-safe scalar down weighted-sum (caller-owned out, no alloc)");
  m.def(
      "active_moe_scalar_out_reference",
      &lynn_native_active_moe_scalar_out_reference,
      "P145 graph-safe scalar active MoE V3.2 (caller-owned scratch, exact reference)");
  m.def(
      "dense_fp4xfp8_scalar_reference",
      &lynn_dense_fp4xfp8_scalar_reference,
      "P191 scalar FP4xFP8 dense FFN reference (E4M3 act x E2M1 weight)");
  m.def(
      "dense_fp4xfp8_mma_probe",
      &lynn_dense_fp4xfp8_mma_probe,
      "P191 SM120a FP4 MMA compilation/capability probe");
}
