/**
 * Lynn Engine · P145 Graph-Safe Scalar MoE V3.2 (ordered exact reference)
 *
 * Caller-owned scratch variant of the exact scalar reference kernels.
 * Uses native FP4→FP32 dequant on-the-fly, FP32 accumulation, BF16 output.
 * NO pre-dequantization to BF16, NO cuBLAS matmuls.
 * Route weights kept in FP32 (no BF16 truncation).
 *
 * Designed to match Triton active / slot PyTorch per-slot order exactly.
 * Slower than V3.1 cuBLAS path, but exact against Triton reference.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHidden = 2048;
constexpr int kIntermediate = 512;
constexpr int kGateUpRows = kIntermediate * 2;
constexpr int kThreads = 256;

__device__ __forceinline__ float e2m1_from_nibble(unsigned char nibble) {
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

// ─────────────────────────────────────────────────────────────────────────────
// Gate/Up SiLU scalar kernel (exact reference, writes to caller-owned buffer)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void gate_up_silu_scalar_out_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ expert_ids,
    const uint8_t* __restrict__ gate_up_packed,
    const float* __restrict__ gate_up_scale,
    const float* __restrict__ gate_up_global_scale,
    __nv_bfloat16* __restrict__ out,
    int64_t packed_stride_e,
    int64_t packed_stride_m,
    int64_t packed_stride_n,
    int64_t scale_stride_e,
    int64_t scale_stride_m,
    int64_t scale_stride_g,
    int64_t out_stride_k,
    int64_t out_stride_i) {
  const int slot = blockIdx.x;
  const int inter = blockIdx.y;
  const int tid = threadIdx.x;
  const int expert = expert_ids[slot];
  const float inv_global = 1.0f / gate_up_global_scale[0];

  extern __shared__ float smem[];
  float gate_acc = 0.0f;
  float up_acc = 0.0f;

  for (int hidden = tid; hidden < kHidden; hidden += blockDim.x) {
    const int packed_col = hidden >> 1;
    const int scale_col = hidden >> 4;
    const bool low = (hidden & 1) == 0;
    const float x_val = __bfloat162float(x[hidden]);
    const uint8_t byte_gate = gate_up_packed[
        static_cast<int64_t>(expert) * packed_stride_e +
        static_cast<int64_t>(inter) * packed_stride_m +
        static_cast<int64_t>(packed_col) * packed_stride_n];
    const uint8_t byte_up = gate_up_packed[
        static_cast<int64_t>(expert) * packed_stride_e +
        static_cast<int64_t>(inter + kIntermediate) * packed_stride_m +
        static_cast<int64_t>(packed_col) * packed_stride_n];
    const unsigned char nibble_gate = low ? (byte_gate & 0x0F) : ((byte_gate >> 4) & 0x0F);
    const unsigned char nibble_up = low ? (byte_up & 0x0F) : ((byte_up >> 4) & 0x0F);
    const float scale_gate = gate_up_scale[
        static_cast<int64_t>(expert) * scale_stride_e +
        static_cast<int64_t>(inter) * scale_stride_m +
        static_cast<int64_t>(scale_col) * scale_stride_g];
    const float scale_up = gate_up_scale[
        static_cast<int64_t>(expert) * scale_stride_e +
        static_cast<int64_t>(inter + kIntermediate) * scale_stride_m +
        static_cast<int64_t>(scale_col) * scale_stride_g];
    gate_acc += x_val * e2m1_from_nibble(nibble_gate) * scale_gate * inv_global;
    up_acc += x_val * e2m1_from_nibble(nibble_up) * scale_up * inv_global;
  }

  smem[tid] = gate_acc;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) smem[tid] += smem[tid + stride];
    __syncthreads();
  }
  const float gate = smem[0];

  smem[tid] = up_acc;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) smem[tid] += smem[tid + stride];
    __syncthreads();
  }
  const float up = smem[0];

  const float silu = gate / (1.0f + expf(-gate));
  out[slot * out_stride_k + inter * out_stride_i] =
      __float2bfloat16(silu * up);
}

// ─────────────────────────────────────────────────────────────────────────────
// Down weighted sum scalar kernel (exact reference, writes to caller-owned buffer)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void down_weighted_sum_scalar_out_kernel(
    const __nv_bfloat16* __restrict__ inter,
    const int32_t* __restrict__ expert_ids,
    const float* __restrict__ routing_weights,
    const uint8_t* __restrict__ down_packed,
    const float* __restrict__ down_scale,
    const float* __restrict__ down_global_scale,
    __nv_bfloat16* __restrict__ out,
    int64_t inter_stride_k,
    int64_t inter_stride_i,
    int64_t packed_stride_e,
    int64_t packed_stride_m,
    int64_t packed_stride_n,
    int64_t scale_stride_e,
    int64_t scale_stride_m,
    int64_t scale_stride_g,
    int64_t top_k) {
  const int hidden = blockIdx.x;
  const int tid = threadIdx.x;
  const float inv_global = 1.0f / down_global_scale[0];

  extern __shared__ float smem[];
  float acc = 0.0f;

  for (int slot = 0; slot < top_k; ++slot) {
    const int expert = expert_ids[slot];
    const float route = routing_weights[slot];
    for (int i = tid; i < kIntermediate; i += blockDim.x) {
      const int packed_col = i >> 1;
      const int scale_col = i >> 4;
      const bool low = (i & 1) == 0;
      const float inter_i = __bfloat162float(
          inter[static_cast<int64_t>(slot) * inter_stride_k +
                static_cast<int64_t>(i) * inter_stride_i]);
      const uint8_t byte = down_packed[
          static_cast<int64_t>(expert) * packed_stride_e +
          static_cast<int64_t>(hidden) * packed_stride_m +
          static_cast<int64_t>(packed_col) * packed_stride_n];
      const unsigned char nibble = low ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
      const float scale = down_scale[
          static_cast<int64_t>(expert) * scale_stride_e +
          static_cast<int64_t>(hidden) * scale_stride_m +
          static_cast<int64_t>(scale_col) * scale_stride_g];
      acc += route * e2m1_from_nibble(nibble) * scale * inv_global * inter_i;
    }
  }

  smem[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) smem[tid] += smem[tid + stride];
    __syncthreads();
  }

  if (tid == 0) {
    out[hidden] = __float2bfloat16(smem[0]);
  }
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Python entry points: caller-owned scratch, zero allocation
// ─────────────────────────────────────────────────────────────────────────────

void lynn_native_gate_up_silu_scalar_out(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    torch::Tensor out) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be a CUDA tensor");
  TORCH_CHECK(gate_up_packed.is_cuda(), "gate_up_packed must be a CUDA tensor");
  TORCH_CHECK(gate_up_scale.is_cuda(), "gate_up_scale must be a CUDA tensor");
  TORCH_CHECK(gate_up_global_scale.is_cuda(), "gate_up_global_scale must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(gate_up_packed.scalar_type() == torch::kUInt8, "gate_up_packed must be uint8");
  TORCH_CHECK(gate_up_scale.scalar_type() == torch::kFloat32, "gate_up_scale must be float32");
  TORCH_CHECK(gate_up_global_scale.scalar_type() == torch::kFloat32, "gate_up_global_scale must be float32");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bfloat16");
  TORCH_CHECK(x.dim() == 1 && x.numel() == kHidden, "x must be [2048]");
  TORCH_CHECK(gate_up_packed.dim() == 3, "gate_up_packed must be [experts, 1024, 1024]");
  TORCH_CHECK(gate_up_scale.dim() == 3, "gate_up_scale must be [experts, 1024, 128]");

  auto xc = x.contiguous();
  auto expert_ids_c = expert_ids.contiguous();
  auto packed_c = gate_up_packed.contiguous();
  auto scale_c = gate_up_scale.contiguous();
  auto global_c = gate_up_global_scale.contiguous();
  auto out_c = out.contiguous();
  const int64_t top_k = expert_ids_c.numel();

  TORCH_CHECK(out_c.dim() == 2 && out_c.size(0) == top_k && out_c.size(1) == kIntermediate,
              "out must be [top_k, 512]");

  const dim3 grid(static_cast<unsigned int>(top_k), kIntermediate);
  const size_t shared_bytes = static_cast<size_t>(kThreads) * 2 * sizeof(float);
  gate_up_silu_scalar_out_kernel<<<grid, kThreads, shared_bytes>>>(
      reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr<at::BFloat16>()),
      expert_ids_c.data_ptr<int32_t>(),
      packed_c.data_ptr<uint8_t>(),
      scale_c.data_ptr<float>(),
      global_c.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out_c.data_ptr<at::BFloat16>()),
      packed_c.stride(0), packed_c.stride(1), packed_c.stride(2),
      scale_c.stride(0), scale_c.stride(1), scale_c.stride(2),
      out_c.stride(0), out_c.stride(1));
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "gate_up_silu_scalar_kernel launch failed: ", cudaGetErrorString(err));
}

void lynn_native_down_weighted_sum_scalar_out(
    torch::Tensor inter,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale,
    torch::Tensor out) {
  TORCH_CHECK(inter.is_cuda(), "inter must be a CUDA tensor");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be a CUDA tensor");
  TORCH_CHECK(routing_weights.is_cuda(), "routing_weights must be a CUDA tensor");
  TORCH_CHECK(down_packed.is_cuda(), "down_packed must be a CUDA tensor");
  TORCH_CHECK(down_scale.is_cuda(), "down_scale must be a CUDA tensor");
  TORCH_CHECK(down_global_scale.is_cuda(), "down_global_scale must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(inter.scalar_type() == torch::kBFloat16, "inter must be bfloat16");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(routing_weights.scalar_type() == torch::kFloat32, "routing_weights must be float32");
  TORCH_CHECK(down_packed.scalar_type() == torch::kUInt8, "down_packed must be uint8");
  TORCH_CHECK(down_scale.scalar_type() == torch::kFloat32, "down_scale must be float32");
  TORCH_CHECK(down_global_scale.scalar_type() == torch::kFloat32, "down_global_scale must be float32");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bfloat16");
  TORCH_CHECK(inter.dim() == 2 && inter.size(1) == kIntermediate, "inter must be [top_k, 512]");
  TORCH_CHECK(down_packed.dim() == 3, "down_packed must be [experts, 2048, 256]");
  TORCH_CHECK(down_scale.dim() == 3, "down_scale must be [experts, 2048, 32]");
  TORCH_CHECK(out.dim() == 1 && out.numel() == kHidden, "out must be [2048]");

  auto inter_c = inter.contiguous();
  auto expert_ids_c = expert_ids.contiguous();
  auto routing_weights_c = routing_weights.contiguous();
  auto down_packed_c = down_packed.contiguous();
  auto down_scale_c = down_scale.contiguous();
  auto down_global_c = down_global_scale.contiguous();
  auto out_c = out.contiguous();
  const int64_t top_k = expert_ids_c.numel();

  TORCH_CHECK(inter_c.size(0) == top_k, "inter batch must match top_k");
  TORCH_CHECK(routing_weights_c.numel() == top_k, "routing_weights must match top_k");

  const size_t shared_bytes = static_cast<size_t>(kThreads) * sizeof(float);
  down_weighted_sum_scalar_out_kernel<<<kHidden, kThreads, shared_bytes>>>(
      reinterpret_cast<const __nv_bfloat16*>(inter_c.data_ptr<at::BFloat16>()),
      expert_ids_c.data_ptr<int32_t>(),
      routing_weights_c.data_ptr<float>(),
      down_packed_c.data_ptr<uint8_t>(),
      down_scale_c.data_ptr<float>(),
      down_global_c.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out_c.data_ptr<at::BFloat16>()),
      inter_c.stride(0), inter_c.stride(1),
      down_packed_c.stride(0), down_packed_c.stride(1), down_packed_c.stride(2),
      down_scale_c.stride(0), down_scale_c.stride(1), down_scale_c.stride(2),
      top_k);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "down_weighted_sum_scalar_kernel launch failed: ", cudaGetErrorString(err));
}

void lynn_native_active_moe_scalar_out_reference(
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
    torch::Tensor out) {
  lynn_native_gate_up_silu_scalar_out(
      x, expert_ids, gate_up_packed, gate_up_scale, gate_up_global_scale, inter_out);
  lynn_native_down_weighted_sum_scalar_out(
      inter_out, expert_ids, routing_weights, down_packed, down_scale, down_global_scale, out);
}
