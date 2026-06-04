#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdlib>

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

namespace {

constexpr int64_t kHidden = 2048;
constexpr int64_t kIntermediate = 512;
constexpr int64_t kGateUpRows = kIntermediate * 2;
constexpr int kP4BThreads = 256;
constexpr int kP4BTopK = 8;

void check_cuda_tensor(const torch::Tensor& t, const char* name, c10::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has wrong dtype");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous for the P4 hot-path ABI");
}

__device__ __forceinline__ float p4b_e2m1_from_nibble(unsigned char nibble) {
  const unsigned char mag = nibble & 0x07;
  const bool sign = (nibble & 0x08) != 0;
  float v = 0.0f;
  if (mag == 1) {
    v = 0.5f;
  } else if (mag == 2) {
    v = 1.0f;
  } else if (mag == 3) {
    v = 1.5f;
  } else if (mag == 4) {
    v = 2.0f;
  } else if (mag == 5) {
    v = 3.0f;
  } else if (mag == 6) {
    v = 4.0f;
  } else if (mag == 7) {
    v = 6.0f;
  }
  return sign ? -v : v;
}

__global__ void p4b_single_cta_reference_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const int32_t* __restrict__ expert_ids,
    const float* __restrict__ routing_weights,
    const uint8_t* __restrict__ gate_up_packed,
    const float* __restrict__ gate_up_scale,
    const float* __restrict__ gate_up_global_scale,
    const uint8_t* __restrict__ down_packed,
    const float* __restrict__ down_scale,
    const float* __restrict__ down_global_scale,
    __nv_bfloat16* __restrict__ out,
    int64_t gate_packed_stride_e,
    int64_t gate_packed_stride_m,
    int64_t gate_packed_stride_n,
    int64_t gate_scale_stride_e,
    int64_t gate_scale_stride_m,
    int64_t gate_scale_stride_g,
    int64_t down_packed_stride_e,
    int64_t down_packed_stride_m,
    int64_t down_packed_stride_n,
    int64_t down_scale_stride_e,
    int64_t down_scale_stride_m,
    int64_t down_scale_stride_g) {
  const int tid = threadIdx.x;
  __shared__ __nv_bfloat16 active[kP4BTopK * kIntermediate];
  __shared__ float reduce_gate[kP4BThreads];
  __shared__ float reduce_up[kP4BThreads];
  __shared__ float reduce_down[kP4BThreads];
  const float gate_inv_global = 1.0f / gate_up_global_scale[0];
  const float down_inv_global = 1.0f / down_global_scale[0];

  for (int slot = 0; slot < kP4BTopK; ++slot) {
    const int expert = expert_ids[slot];
    for (int inter = 0; inter < kIntermediate; ++inter) {
      float gate_acc = 0.0f;
      float up_acc = 0.0f;
      for (int h = tid; h < kHidden; h += blockDim.x) {
        const int packed_col = h >> 1;
        const int scale_col = h >> 4;
        const bool low = (h & 1) == 0;
        const float x_h = __bfloat162float(hidden[h]);
        const uint8_t gate_byte = gate_up_packed[
            static_cast<int64_t>(expert) * gate_packed_stride_e +
            static_cast<int64_t>(inter) * gate_packed_stride_m +
            static_cast<int64_t>(packed_col) * gate_packed_stride_n];
        const uint8_t up_byte = gate_up_packed[
            static_cast<int64_t>(expert) * gate_packed_stride_e +
            static_cast<int64_t>(inter + kIntermediate) * gate_packed_stride_m +
            static_cast<int64_t>(packed_col) * gate_packed_stride_n];
        const unsigned char gate_nibble = low ? (gate_byte & 0x0F) : ((gate_byte >> 4) & 0x0F);
        const unsigned char up_nibble = low ? (up_byte & 0x0F) : ((up_byte >> 4) & 0x0F);
        const float gate_scale = gate_up_scale[
            static_cast<int64_t>(expert) * gate_scale_stride_e +
            static_cast<int64_t>(inter) * gate_scale_stride_m +
            static_cast<int64_t>(scale_col) * gate_scale_stride_g];
        const float up_scale = gate_up_scale[
            static_cast<int64_t>(expert) * gate_scale_stride_e +
            static_cast<int64_t>(inter + kIntermediate) * gate_scale_stride_m +
            static_cast<int64_t>(scale_col) * gate_scale_stride_g];
        gate_acc += x_h * p4b_e2m1_from_nibble(gate_nibble) * gate_scale * gate_inv_global;
        up_acc += x_h * p4b_e2m1_from_nibble(up_nibble) * up_scale * gate_inv_global;
      }

      reduce_gate[tid] = gate_acc;
      reduce_up[tid] = up_acc;
      __syncthreads();
      for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
          reduce_gate[tid] += reduce_gate[tid + stride];
          reduce_up[tid] += reduce_up[tid + stride];
        }
        __syncthreads();
      }
      if (tid == 0) {
        const float gate = reduce_gate[0];
        const float up = reduce_up[0];
        const float silu = gate / (1.0f + expf(-gate));
        active[slot * kIntermediate + inter] = __float2bfloat16(silu * up);
      }
      __syncthreads();
    }
  }

  for (int h = 0; h < kHidden; ++h) {
    float partial = 0.0f;
    for (int slot = 0; slot < kP4BTopK; ++slot) {
      const int expert = expert_ids[slot];
      const float route = routing_weights[slot];
      for (int inter = tid; inter < kIntermediate; inter += blockDim.x) {
        const int packed_col = inter >> 1;
        const int scale_col = inter >> 4;
        const bool low = (inter & 1) == 0;
        const uint8_t byte = down_packed[
            static_cast<int64_t>(expert) * down_packed_stride_e +
            static_cast<int64_t>(h) * down_packed_stride_m +
            static_cast<int64_t>(packed_col) * down_packed_stride_n];
        const unsigned char nibble = low ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        const float scale = down_scale[
            static_cast<int64_t>(expert) * down_scale_stride_e +
            static_cast<int64_t>(h) * down_scale_stride_m +
            static_cast<int64_t>(scale_col) * down_scale_stride_g];
        const float active_value = __bfloat162float(active[slot * kIntermediate + inter]);
        partial += route * p4b_e2m1_from_nibble(nibble) * scale * down_inv_global * active_value;
      }
    }
    reduce_down[tid] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce_down[tid] += reduce_down[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      out[h] = __float2bfloat16(reduce_down[0]);
    }
    __syncthreads();
  }
}

__global__ void p4b_multi_cta_reference_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const int32_t* __restrict__ expert_ids,
    const float* __restrict__ routing_weights,
    const uint8_t* __restrict__ gate_up_packed,
    const float* __restrict__ gate_up_scale,
    const float* __restrict__ gate_up_global_scale,
    const uint8_t* __restrict__ down_packed,
    const float* __restrict__ down_scale,
    const float* __restrict__ down_global_scale,
    __nv_bfloat16* __restrict__ out,
    int64_t gate_packed_stride_e,
    int64_t gate_packed_stride_m,
    int64_t gate_packed_stride_n,
    int64_t gate_scale_stride_e,
    int64_t gate_scale_stride_m,
    int64_t gate_scale_stride_g,
    int64_t down_packed_stride_e,
    int64_t down_packed_stride_m,
    int64_t down_packed_stride_n,
    int64_t down_scale_stride_e,
    int64_t down_scale_stride_m,
    int64_t down_scale_stride_g,
    int tile_hidden) {
  const int tid = threadIdx.x;
  const int h_begin = blockIdx.x * tile_hidden;
  const int h_end = min(h_begin + tile_hidden, static_cast<int>(kHidden));
  __shared__ __nv_bfloat16 active[kP4BTopK * kIntermediate];
  __shared__ float reduce_gate[kP4BThreads];
  __shared__ float reduce_up[kP4BThreads];
  __shared__ float reduce_down[kP4BThreads];
  const float gate_inv_global = 1.0f / gate_up_global_scale[0];
  const float down_inv_global = 1.0f / down_global_scale[0];

  // Correctness-first parallel reference: each CTA owns a hidden-output tile.
  // It recomputes active[slot, inter] per tile to avoid any inter_scratch ABI.
  for (int slot = 0; slot < kP4BTopK; ++slot) {
    const int expert = expert_ids[slot];
    for (int inter = 0; inter < kIntermediate; ++inter) {
      float gate_acc = 0.0f;
      float up_acc = 0.0f;
      for (int h = tid; h < kHidden; h += blockDim.x) {
        const int packed_col = h >> 1;
        const int scale_col = h >> 4;
        const bool low = (h & 1) == 0;
        const float x_h = __bfloat162float(hidden[h]);
        const uint8_t gate_byte = gate_up_packed[
            static_cast<int64_t>(expert) * gate_packed_stride_e +
            static_cast<int64_t>(inter) * gate_packed_stride_m +
            static_cast<int64_t>(packed_col) * gate_packed_stride_n];
        const uint8_t up_byte = gate_up_packed[
            static_cast<int64_t>(expert) * gate_packed_stride_e +
            static_cast<int64_t>(inter + kIntermediate) * gate_packed_stride_m +
            static_cast<int64_t>(packed_col) * gate_packed_stride_n];
        const unsigned char gate_nibble = low ? (gate_byte & 0x0F) : ((gate_byte >> 4) & 0x0F);
        const unsigned char up_nibble = low ? (up_byte & 0x0F) : ((up_byte >> 4) & 0x0F);
        const float gate_scale = gate_up_scale[
            static_cast<int64_t>(expert) * gate_scale_stride_e +
            static_cast<int64_t>(inter) * gate_scale_stride_m +
            static_cast<int64_t>(scale_col) * gate_scale_stride_g];
        const float up_scale = gate_up_scale[
            static_cast<int64_t>(expert) * gate_scale_stride_e +
            static_cast<int64_t>(inter + kIntermediate) * gate_scale_stride_m +
            static_cast<int64_t>(scale_col) * gate_scale_stride_g];
        gate_acc += x_h * p4b_e2m1_from_nibble(gate_nibble) * gate_scale * gate_inv_global;
        up_acc += x_h * p4b_e2m1_from_nibble(up_nibble) * up_scale * gate_inv_global;
      }

      reduce_gate[tid] = gate_acc;
      reduce_up[tid] = up_acc;
      __syncthreads();
      for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
          reduce_gate[tid] += reduce_gate[tid + stride];
          reduce_up[tid] += reduce_up[tid + stride];
        }
        __syncthreads();
      }
      if (tid == 0) {
        const float gate = reduce_gate[0];
        const float up = reduce_up[0];
        const float silu = gate / (1.0f + expf(-gate));
        active[slot * kIntermediate + inter] = __float2bfloat16(silu * up);
      }
      __syncthreads();
    }
  }

  for (int h = h_begin; h < h_end; ++h) {
    float partial = 0.0f;
    for (int slot = 0; slot < kP4BTopK; ++slot) {
      const int expert = expert_ids[slot];
      const float route = routing_weights[slot];
      for (int inter = tid; inter < kIntermediate; inter += blockDim.x) {
        const int packed_col = inter >> 1;
        const int scale_col = inter >> 4;
        const bool low = (inter & 1) == 0;
        const uint8_t byte = down_packed[
            static_cast<int64_t>(expert) * down_packed_stride_e +
            static_cast<int64_t>(h) * down_packed_stride_m +
            static_cast<int64_t>(packed_col) * down_packed_stride_n];
        const unsigned char nibble = low ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        const float scale = down_scale[
            static_cast<int64_t>(expert) * down_scale_stride_e +
            static_cast<int64_t>(h) * down_scale_stride_m +
            static_cast<int64_t>(scale_col) * down_scale_stride_g];
        const float active_value = __bfloat162float(active[slot * kIntermediate + inter]);
        partial += route * p4b_e2m1_from_nibble(nibble) * scale * down_inv_global * active_value;
      }
    }
    reduce_down[tid] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce_down[tid] += reduce_down[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      out[h] = __float2bfloat16(reduce_down[0]);
    }
    __syncthreads();
  }
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
    torch::Tensor inter_scratch,
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
  check_cuda_tensor(inter_scratch, "inter_scratch", torch::kBFloat16);
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
  const int64_t top_k = expert_ids.size(1);
  TORCH_CHECK(
      inter_scratch.dim() == 3 && inter_scratch.size(0) == tokens &&
          inter_scratch.size(1) == top_k && inter_scratch.size(2) == kIntermediate,
      "inter_scratch must be [T, top_k, 512]");

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
  TORCH_CHECK(tokens == 1, "P4 two-stage reference currently supports T=1 decode only");
  TORCH_CHECK(tile_tokens == 1, "P4 two-stage reference currently supports tile_tokens=1 only");
  TORCH_CHECK(tile_inter == 1 || tile_inter == 2 || tile_inter == 4 || tile_inter == 8,
              "tile_inter must be one of {1, 2, 4, 8}");
  TORCH_CHECK(tile_hidden == 1 || tile_hidden == 2 || tile_hidden == 4 || tile_hidden == 8,
              "tile_hidden must be one of {1, 2, 4, 8}");

  auto hidden_1d = hidden.view({kHidden});
  auto expert_ids_1d = expert_ids.view({top_k});
  auto routing_weights_1d = routing_weights.view({top_k});
  auto inter_2d = inter_scratch.view({top_k, kIntermediate});
  auto out_1d = out.view({kHidden});
  lynn_native_active_moe_grouped_per16_nonatomic_out_reference(
      hidden_1d,
      expert_ids_1d,
      routing_weights_1d,
      gate_up_packed,
      gate_up_scale,
      gate_up_global_scale,
      down_packed,
      down_scale,
      down_global_scale,
      inter_2d,
      out_1d,
      tile_inter,
      tile_hidden);
}

void lynn_native_active_moe_fused_zero_shadow_single_kernel_contract(
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
    int64_t tile_experts,
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

  TORCH_CHECK(hidden.dim() == 2, "P4B hidden must be [T, 2048]");
  const int64_t tokens = hidden.size(0);
  TORCH_CHECK(tokens > 0, "P4B hidden token dimension must be non-empty");
  TORCH_CHECK(hidden.size(1) == kHidden, "P4B hidden dim must be 2048");
  TORCH_CHECK(out.dim() == 2 && out.size(0) == tokens && out.size(1) == kHidden, "P4B out must be [T, 2048]");

  TORCH_CHECK(expert_ids.dim() == 2, "P4B expert_ids must be [T, top_k]");
  TORCH_CHECK(routing_weights.dim() == 2, "P4B routing_weights must be [T, top_k]");
  TORCH_CHECK(expert_ids.size(0) == tokens, "P4B expert_ids token dimension must match hidden");
  TORCH_CHECK(routing_weights.size(0) == tokens, "P4B routing_weights token dimension must match hidden");
  TORCH_CHECK(routing_weights.size(1) == expert_ids.size(1), "P4B routing_weights top_k must match expert_ids");
  TORCH_CHECK(expert_ids.size(1) > 0, "P4B top_k must be non-empty");

  TORCH_CHECK(gate_up_packed.dim() == 3, "P4B gate_up_packed must be [E, 1024, 1024]");
  TORCH_CHECK(gate_up_scale.dim() == 3, "P4B gate_up_scale must be [E, 1024, 128]");
  TORCH_CHECK(down_packed.dim() == 3, "P4B down_packed must be [E, 2048, 256]");
  TORCH_CHECK(down_scale.dim() == 3, "P4B down_scale must be [E, 2048, 32]");
  const int64_t experts = gate_up_packed.size(0);
  TORCH_CHECK(experts > 0, "P4B expert dimension must be non-empty");
  TORCH_CHECK(gate_up_scale.size(0) == experts, "P4B gate_up_scale expert dimension must match gate_up_packed");
  TORCH_CHECK(down_packed.size(0) == experts, "P4B down_packed expert dimension must match gate_up_packed");
  TORCH_CHECK(down_scale.size(0) == experts, "P4B down_scale expert dimension must match gate_up_packed");
  TORCH_CHECK(gate_up_packed.size(1) == kGateUpRows, "P4B gate_up_packed row dim must be 1024");
  TORCH_CHECK(gate_up_packed.size(2) == kHidden / 2, "P4B gate_up_packed packed hidden dim must be 1024");
  TORCH_CHECK(gate_up_scale.size(1) == kGateUpRows, "P4B gate_up_scale row dim must be 1024");
  TORCH_CHECK(gate_up_scale.size(2) == kHidden / 16, "P4B gate_up_scale group dim must be 128");
  TORCH_CHECK(down_packed.size(1) == kHidden, "P4B down_packed row dim must be 2048");
  TORCH_CHECK(down_packed.size(2) == kIntermediate / 2, "P4B down_packed packed inter dim must be 256");
  TORCH_CHECK(down_scale.size(1) == kHidden, "P4B down_scale row dim must be 2048");
  TORCH_CHECK(down_scale.size(2) == kIntermediate / 16, "P4B down_scale group dim must be 32");
  TORCH_CHECK(gate_up_global_scale.numel() == 1, "P4B gate_up_global_scale must be scalar");
  TORCH_CHECK(down_global_scale.numel() == 1, "P4B down_global_scale must be scalar");
  TORCH_CHECK(tile_tokens > 0, "P4B tile_tokens must be positive");
  TORCH_CHECK(tile_experts > 0, "P4B tile_experts must be positive");
  TORCH_CHECK(tile_hidden > 0, "P4B tile_hidden must be positive");

  const char* single_cta_reference = std::getenv("LYNN_NATIVE_P4B_SINGLE_CTA_REFERENCE");
  const char* multi_cta_reference = std::getenv("LYNN_NATIVE_P4B_MULTI_CTA_REFERENCE");
  if (multi_cta_reference != nullptr && multi_cta_reference[0] == '1' && multi_cta_reference[1] == '\0') {
    TORCH_CHECK(tokens == 1, "P4B multi-CTA reference currently supports T=1 decode only");
    TORCH_CHECK(expert_ids.size(1) == kP4BTopK, "P4B multi-CTA reference currently supports top_k=8 only");
    TORCH_CHECK(tile_tokens == 1, "P4B multi-CTA reference currently supports tile_tokens=1 only");
    TORCH_CHECK(tile_experts == 1, "P4B multi-CTA reference currently supports tile_experts=1 only");
    TORCH_CHECK(tile_hidden > 0 && tile_hidden <= kHidden, "P4B multi-CTA reference tile_hidden must be in [1, 2048]");
    const int grid_x = static_cast<int>((kHidden + tile_hidden - 1) / tile_hidden);
    p4b_multi_cta_reference_kernel<<<grid_x, kP4BThreads>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()),
        expert_ids.data_ptr<int32_t>(),
        routing_weights.data_ptr<float>(),
        gate_up_packed.data_ptr<uint8_t>(),
        gate_up_scale.data_ptr<float>(),
        gate_up_global_scale.data_ptr<float>(),
        down_packed.data_ptr<uint8_t>(),
        down_scale.data_ptr<float>(),
        down_global_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        gate_up_packed.stride(0),
        gate_up_packed.stride(1),
        gate_up_packed.stride(2),
        gate_up_scale.stride(0),
        gate_up_scale.stride(1),
        gate_up_scale.stride(2),
        down_packed.stride(0),
        down_packed.stride(1),
        down_packed.stride(2),
        down_scale.stride(0),
        down_scale.stride(1),
        down_scale.stride(2),
        static_cast<int>(tile_hidden));
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "P4B multi-CTA reference launch failed: ", cudaGetErrorString(err));
    return;
  }

  if (single_cta_reference != nullptr && single_cta_reference[0] == '1' && single_cta_reference[1] == '\0') {
    TORCH_CHECK(tokens == 1, "P4B single-CTA reference currently supports T=1 decode only");
    TORCH_CHECK(expert_ids.size(1) == kP4BTopK, "P4B single-CTA reference currently supports top_k=8 only");
    TORCH_CHECK(tile_tokens == 1, "P4B single-CTA reference currently supports tile_tokens=1 only");
    TORCH_CHECK(tile_experts == 1, "P4B single-CTA reference currently supports tile_experts=1 only");
    TORCH_CHECK(tile_hidden == 8, "P4B single-CTA reference currently supports tile_hidden=8 only");

    p4b_single_cta_reference_kernel<<<1, kP4BThreads>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()),
        expert_ids.data_ptr<int32_t>(),
        routing_weights.data_ptr<float>(),
        gate_up_packed.data_ptr<uint8_t>(),
        gate_up_scale.data_ptr<float>(),
        gate_up_global_scale.data_ptr<float>(),
        down_packed.data_ptr<uint8_t>(),
        down_scale.data_ptr<float>(),
        down_global_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        gate_up_packed.stride(0),
        gate_up_packed.stride(1),
        gate_up_packed.stride(2),
        gate_up_scale.stride(0),
        gate_up_scale.stride(1),
        gate_up_scale.stride(2),
        down_packed.stride(0),
        down_packed.stride(1),
        down_packed.stride(2),
        down_scale.stride(0),
        down_scale.stride(1),
        down_scale.stride(2));
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "P4B single-CTA reference launch failed: ", cudaGetErrorString(err));
    return;
  }

  TORCH_CHECK(
      false,
      "P4B single-kernel fused zero-shadow contract is not implemented yet; "
      "do not bank fused-kernel speed or promote this backend");
}
