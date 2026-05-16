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

__global__ void gate_up_silu_scalar_kernel(
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
  const int gate_row = inter;
  const int up_row = kIntermediate + inter;
  const float inv_global = 1.0f / gate_up_global_scale[0];

  extern __shared__ float smem[];
  float* gate_s = smem;
  float* up_s = smem + blockDim.x;

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

  for (int h = tid; h < kHidden; h += blockDim.x) {
    const int packed_col = h >> 1;
    const int scale_col = h >> 4;
    const bool low = (h & 1) == 0;
    const float x_h = __bfloat162float(x[h]);

    const uint8_t gate_byte = gate_up_packed[
        static_cast<int64_t>(expert) * packed_stride_e +
        static_cast<int64_t>(gate_row) * packed_stride_m +
        static_cast<int64_t>(packed_col) * packed_stride_n];
    const uint8_t up_byte = gate_up_packed[
        static_cast<int64_t>(expert) * packed_stride_e +
        static_cast<int64_t>(up_row) * packed_stride_m +
        static_cast<int64_t>(packed_col) * packed_stride_n];
    const unsigned char gate_nibble = low ? (gate_byte & 0x0F) : ((gate_byte >> 4) & 0x0F);
    const unsigned char up_nibble = low ? (up_byte & 0x0F) : ((up_byte >> 4) & 0x0F);

    const float gate_scale = gate_up_scale[
        static_cast<int64_t>(expert) * scale_stride_e +
        static_cast<int64_t>(gate_row) * scale_stride_m +
        static_cast<int64_t>(scale_col) * scale_stride_g];
    const float up_scale = gate_up_scale[
        static_cast<int64_t>(expert) * scale_stride_e +
        static_cast<int64_t>(up_row) * scale_stride_m +
        static_cast<int64_t>(scale_col) * scale_stride_g];

    gate_acc += e2m1_from_nibble(gate_nibble) * gate_scale * inv_global * x_h;
    up_acc += e2m1_from_nibble(up_nibble) * up_scale * inv_global * x_h;
  }

  gate_s[tid] = gate_acc;
  up_s[tid] = up_acc;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      gate_s[tid] += gate_s[tid + stride];
      up_s[tid] += up_s[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float gate = gate_s[0];
    const float up = up_s[0];
    const float silu = gate / (1.0f + expf(-gate));
    out[static_cast<int64_t>(slot) * out_stride_k + static_cast<int64_t>(inter) * out_stride_i] =
        __float2bfloat16(silu * up);
  }
}

template <int TILE_INTER, int THREADS>
__global__ void gate_up_silu_tile_inter_scalar_kernel(
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
  const int inter_start = blockIdx.y * TILE_INTER;
  const int tid = threadIdx.x;
  const int expert = expert_ids[slot];
  const float inv_global = 1.0f / gate_up_global_scale[0];

  float gate_acc[TILE_INTER];
  float up_acc[TILE_INTER];
#pragma unroll
  for (int r = 0; r < TILE_INTER; ++r) {
    gate_acc[r] = 0.0f;
    up_acc[r] = 0.0f;
  }

  for (int h = tid; h < kHidden; h += THREADS) {
    const int packed_col = h >> 1;
    const int scale_col = h >> 4;
    const bool low = (h & 1) == 0;
    const float x_h = __bfloat162float(x[h]);

#pragma unroll
    for (int r = 0; r < TILE_INTER; ++r) {
      const int inter = inter_start + r;
      if (inter < kIntermediate) {
        const int gate_row = inter;
        const int up_row = kIntermediate + inter;
        const uint8_t gate_byte = gate_up_packed[
            static_cast<int64_t>(expert) * packed_stride_e +
            static_cast<int64_t>(gate_row) * packed_stride_m +
            static_cast<int64_t>(packed_col) * packed_stride_n];
        const uint8_t up_byte = gate_up_packed[
            static_cast<int64_t>(expert) * packed_stride_e +
            static_cast<int64_t>(up_row) * packed_stride_m +
            static_cast<int64_t>(packed_col) * packed_stride_n];
        const unsigned char gate_nibble = low ? (gate_byte & 0x0F) : ((gate_byte >> 4) & 0x0F);
        const unsigned char up_nibble = low ? (up_byte & 0x0F) : ((up_byte >> 4) & 0x0F);
        const float gate_scale = gate_up_scale[
            static_cast<int64_t>(expert) * scale_stride_e +
            static_cast<int64_t>(gate_row) * scale_stride_m +
            static_cast<int64_t>(scale_col) * scale_stride_g];
        const float up_scale = gate_up_scale[
            static_cast<int64_t>(expert) * scale_stride_e +
            static_cast<int64_t>(up_row) * scale_stride_m +
            static_cast<int64_t>(scale_col) * scale_stride_g];
        gate_acc[r] += e2m1_from_nibble(gate_nibble) * gate_scale * inv_global * x_h;
        up_acc[r] += e2m1_from_nibble(up_nibble) * up_scale * inv_global * x_h;
      }
    }
  }

  extern __shared__ float smem[];
  float* gate_s = smem;
  float* up_s = smem + TILE_INTER * THREADS;

#pragma unroll
  for (int r = 0; r < TILE_INTER; ++r) {
    gate_s[r * THREADS + tid] = gate_acc[r];
    up_s[r * THREADS + tid] = up_acc[r];
  }
  __syncthreads();

  for (int stride = THREADS >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
#pragma unroll
      for (int r = 0; r < TILE_INTER; ++r) {
        gate_s[r * THREADS + tid] += gate_s[r * THREADS + tid + stride];
        up_s[r * THREADS + tid] += up_s[r * THREADS + tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
#pragma unroll
    for (int r = 0; r < TILE_INTER; ++r) {
      const int inter = inter_start + r;
      if (inter < kIntermediate) {
        const float gate = gate_s[r * THREADS];
        const float up = up_s[r * THREADS];
        const float silu = gate / (1.0f + expf(-gate));
        out[static_cast<int64_t>(slot) * out_stride_k + static_cast<int64_t>(inter) * out_stride_i] =
            __float2bfloat16(silu * up);
      }
    }
  }
}

__global__ void down_weighted_sum_scalar_kernel(
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
          inter[static_cast<int64_t>(slot) * inter_stride_k + static_cast<int64_t>(i) * inter_stride_i]);
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
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[hidden] = __float2bfloat16(smem[0]);
  }
}

template <int TILE_HIDDEN, int THREADS_X>
__global__ void down_weighted_sum_tile_scalar_kernel(
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
  const int tile_start = blockIdx.x * TILE_HIDDEN;
  const int local_hidden = threadIdx.y;
  const int hidden = tile_start + local_hidden;
  const int tx = threadIdx.x;
  const float inv_global = 1.0f / down_global_scale[0];

  extern __shared__ float smem[];
  float* row_smem = smem + local_hidden * THREADS_X;
  float acc = 0.0f;

  if (hidden < kHidden) {
    for (int slot = 0; slot < top_k; ++slot) {
      const int expert = expert_ids[slot];
      const float route = routing_weights[slot];
      for (int i = tx; i < kIntermediate; i += THREADS_X) {
        const int packed_col = i >> 1;
        const int scale_col = i >> 4;
        const bool low = (i & 1) == 0;
        const float inter_i = __bfloat162float(
            inter[static_cast<int64_t>(slot) * inter_stride_k + static_cast<int64_t>(i) * inter_stride_i]);
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
  }

  row_smem[tx] = acc;
  __syncthreads();

  for (int stride = THREADS_X >> 1; stride > 0; stride >>= 1) {
    if (tx < stride) {
      row_smem[tx] += row_smem[tx + stride];
    }
    __syncthreads();
  }

  if (tx == 0 && hidden < kHidden) {
    out[hidden] = __float2bfloat16(row_smem[0]);
  }
}

__global__ void active_moe_fused_atomic_scalar_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ expert_ids,
    const float* __restrict__ routing_weights,
    const uint8_t* __restrict__ gate_up_packed,
    const float* __restrict__ gate_up_scale,
    const float* __restrict__ gate_up_global_scale,
    const uint8_t* __restrict__ down_packed,
    const float* __restrict__ down_scale,
    const float* __restrict__ down_global_scale,
    float* __restrict__ out,
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
  const int slot = blockIdx.x;
  const int inter = blockIdx.y;
  const int tid = threadIdx.x;
  const int expert = expert_ids[slot];
  const int gate_row = inter;
  const int up_row = kIntermediate + inter;
  const float gate_inv_global = 1.0f / gate_up_global_scale[0];
  const float down_inv_global = 1.0f / down_global_scale[0];

  extern __shared__ float smem[];
  float* gate_s = smem;
  float* up_s = smem + blockDim.x;

  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (int h = tid; h < kHidden; h += blockDim.x) {
    const int packed_col = h >> 1;
    const int scale_col = h >> 4;
    const bool low = (h & 1) == 0;
    const float x_h = __bfloat162float(x[h]);
    const uint8_t gate_byte = gate_up_packed[
        static_cast<int64_t>(expert) * gate_packed_stride_e +
        static_cast<int64_t>(gate_row) * gate_packed_stride_m +
        static_cast<int64_t>(packed_col) * gate_packed_stride_n];
    const uint8_t up_byte = gate_up_packed[
        static_cast<int64_t>(expert) * gate_packed_stride_e +
        static_cast<int64_t>(up_row) * gate_packed_stride_m +
        static_cast<int64_t>(packed_col) * gate_packed_stride_n];
    const unsigned char gate_nibble = low ? (gate_byte & 0x0F) : ((gate_byte >> 4) & 0x0F);
    const unsigned char up_nibble = low ? (up_byte & 0x0F) : ((up_byte >> 4) & 0x0F);
    const float gate_scale = gate_up_scale[
        static_cast<int64_t>(expert) * gate_scale_stride_e +
        static_cast<int64_t>(gate_row) * gate_scale_stride_m +
        static_cast<int64_t>(scale_col) * gate_scale_stride_g];
    const float up_scale = gate_up_scale[
        static_cast<int64_t>(expert) * gate_scale_stride_e +
        static_cast<int64_t>(up_row) * gate_scale_stride_m +
        static_cast<int64_t>(scale_col) * gate_scale_stride_g];
    gate_acc += e2m1_from_nibble(gate_nibble) * gate_scale * gate_inv_global * x_h;
    up_acc += e2m1_from_nibble(up_nibble) * up_scale * gate_inv_global * x_h;
  }

  gate_s[tid] = gate_acc;
  up_s[tid] = up_acc;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      gate_s[tid] += gate_s[tid + stride];
      up_s[tid] += up_s[tid + stride];
    }
    __syncthreads();
  }

  const float gate = gate_s[0];
  const float up = up_s[0];
  const float silu = gate / (1.0f + expf(-gate));
  const float inter_val = silu * up;
  const float route = routing_weights[slot];

  for (int hidden = tid; hidden < kHidden; hidden += blockDim.x) {
    const int packed_col = inter >> 1;
    const int scale_col = inter >> 4;
    const bool low = (inter & 1) == 0;
    const uint8_t byte = down_packed[
        static_cast<int64_t>(expert) * down_packed_stride_e +
        static_cast<int64_t>(hidden) * down_packed_stride_m +
        static_cast<int64_t>(packed_col) * down_packed_stride_n];
    const unsigned char nibble = low ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
    const float scale = down_scale[
        static_cast<int64_t>(expert) * down_scale_stride_e +
        static_cast<int64_t>(hidden) * down_scale_stride_m +
        static_cast<int64_t>(scale_col) * down_scale_stride_g];
    atomicAdd(out + hidden, route * e2m1_from_nibble(nibble) * scale * down_inv_global * inter_val);
  }
}

}  // namespace

torch::Tensor lynn_native_gate_up_silu_scalar(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be a CUDA tensor");
  TORCH_CHECK(gate_up_packed.is_cuda(), "gate_up_packed must be a CUDA tensor");
  TORCH_CHECK(gate_up_scale.is_cuda(), "gate_up_scale must be a CUDA tensor");
  TORCH_CHECK(gate_up_global_scale.is_cuda(), "gate_up_global_scale must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(gate_up_packed.scalar_type() == torch::kUInt8, "gate_up_packed must be uint8");
  TORCH_CHECK(gate_up_scale.scalar_type() == torch::kFloat32, "gate_up_scale must be float32");
  TORCH_CHECK(gate_up_global_scale.scalar_type() == torch::kFloat32, "gate_up_global_scale must be float32");
  TORCH_CHECK(x.dim() == 1 && x.numel() == kHidden, "x must be [2048]");
  TORCH_CHECK(gate_up_packed.dim() == 3, "gate_up_packed must be [experts, 1024, 1024]");
  TORCH_CHECK(gate_up_scale.dim() == 3, "gate_up_scale must be [experts, 1024, 128]");
  TORCH_CHECK(gate_up_packed.size(1) == kGateUpRows, "gate_up_packed row dim must be 1024");
  TORCH_CHECK(gate_up_packed.size(2) == kHidden / 2, "gate_up_packed packed hidden dim must be 1024");
  TORCH_CHECK(gate_up_scale.size(1) == kGateUpRows, "gate_up_scale row dim must be 1024");
  TORCH_CHECK(gate_up_scale.size(2) == kHidden / 16, "gate_up_scale group dim must be 128");
  TORCH_CHECK(gate_up_global_scale.numel() == 1, "gate_up_global_scale must be scalar");

  auto xc = x.contiguous();
  auto expert_ids_c = expert_ids.contiguous();
  auto packed_c = gate_up_packed.contiguous();
  auto scale_c = gate_up_scale.contiguous();
  auto global_c = gate_up_global_scale.contiguous();
  const int64_t top_k = expert_ids_c.numel();
  auto out = torch::empty({top_k, kIntermediate}, x.options());

  const dim3 grid(static_cast<unsigned int>(top_k), kIntermediate);
  const size_t shared_bytes = static_cast<size_t>(kThreads) * 2 * sizeof(float);
  gate_up_silu_scalar_kernel<<<grid, kThreads, shared_bytes>>>(
      reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr<at::BFloat16>()),
      expert_ids_c.data_ptr<int32_t>(),
      packed_c.data_ptr<uint8_t>(),
      scale_c.data_ptr<float>(),
      global_c.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      packed_c.stride(0),
      packed_c.stride(1),
      packed_c.stride(2),
      scale_c.stride(0),
      scale_c.stride(1),
      scale_c.stride(2),
      out.stride(0),
      out.stride(1));
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "gate_up_silu_scalar_kernel launch failed: ", cudaGetErrorString(err));
  return out;
}

torch::Tensor lynn_native_gate_up_silu_tile_inter_scalar(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    int64_t tile_inter) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be a CUDA tensor");
  TORCH_CHECK(gate_up_packed.is_cuda(), "gate_up_packed must be a CUDA tensor");
  TORCH_CHECK(gate_up_scale.is_cuda(), "gate_up_scale must be a CUDA tensor");
  TORCH_CHECK(gate_up_global_scale.is_cuda(), "gate_up_global_scale must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(gate_up_packed.scalar_type() == torch::kUInt8, "gate_up_packed must be uint8");
  TORCH_CHECK(gate_up_scale.scalar_type() == torch::kFloat32, "gate_up_scale must be float32");
  TORCH_CHECK(gate_up_global_scale.scalar_type() == torch::kFloat32, "gate_up_global_scale must be float32");
  TORCH_CHECK(x.dim() == 1 && x.numel() == kHidden, "x must be [2048]");
  TORCH_CHECK(gate_up_packed.dim() == 3, "gate_up_packed must be [experts, 1024, 1024]");
  TORCH_CHECK(gate_up_scale.dim() == 3, "gate_up_scale must be [experts, 1024, 128]");
  TORCH_CHECK(gate_up_packed.size(1) == kGateUpRows, "gate_up_packed row dim must be 1024");
  TORCH_CHECK(gate_up_packed.size(2) == kHidden / 2, "gate_up_packed packed hidden dim must be 1024");
  TORCH_CHECK(gate_up_scale.size(1) == kGateUpRows, "gate_up_scale row dim must be 1024");
  TORCH_CHECK(gate_up_scale.size(2) == kHidden / 16, "gate_up_scale group dim must be 128");
  TORCH_CHECK(gate_up_global_scale.numel() == 1, "gate_up_global_scale must be scalar");
  TORCH_CHECK(
      tile_inter == 1 || tile_inter == 2 || tile_inter == 4 || tile_inter == 8,
      "tile_inter must be one of {1, 2, 4, 8}");

  auto xc = x.contiguous();
  auto expert_ids_c = expert_ids.contiguous();
  auto packed_c = gate_up_packed.contiguous();
  auto scale_c = gate_up_scale.contiguous();
  auto global_c = gate_up_global_scale.contiguous();
  const int64_t top_k = expert_ids_c.numel();
  auto out = torch::empty({top_k, kIntermediate}, x.options());

  constexpr int kTileThreads = 128;
  const unsigned int grid_y = static_cast<unsigned int>((kIntermediate + tile_inter - 1) / tile_inter);
  const dim3 grid(static_cast<unsigned int>(top_k), grid_y);
  const size_t shared_bytes = static_cast<size_t>(tile_inter) * kTileThreads * 2 * sizeof(float);

  if (tile_inter == 1) {
    gate_up_silu_tile_inter_scalar_kernel<1, kTileThreads><<<grid, kTileThreads, shared_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr<at::BFloat16>()),
        expert_ids_c.data_ptr<int32_t>(),
        packed_c.data_ptr<uint8_t>(),
        scale_c.data_ptr<float>(),
        global_c.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        packed_c.stride(0),
        packed_c.stride(1),
        packed_c.stride(2),
        scale_c.stride(0),
        scale_c.stride(1),
        scale_c.stride(2),
        out.stride(0),
        out.stride(1));
  } else if (tile_inter == 2) {
    gate_up_silu_tile_inter_scalar_kernel<2, kTileThreads><<<grid, kTileThreads, shared_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr<at::BFloat16>()),
        expert_ids_c.data_ptr<int32_t>(),
        packed_c.data_ptr<uint8_t>(),
        scale_c.data_ptr<float>(),
        global_c.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        packed_c.stride(0),
        packed_c.stride(1),
        packed_c.stride(2),
        scale_c.stride(0),
        scale_c.stride(1),
        scale_c.stride(2),
        out.stride(0),
        out.stride(1));
  } else if (tile_inter == 4) {
    gate_up_silu_tile_inter_scalar_kernel<4, kTileThreads><<<grid, kTileThreads, shared_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr<at::BFloat16>()),
        expert_ids_c.data_ptr<int32_t>(),
        packed_c.data_ptr<uint8_t>(),
        scale_c.data_ptr<float>(),
        global_c.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        packed_c.stride(0),
        packed_c.stride(1),
        packed_c.stride(2),
        scale_c.stride(0),
        scale_c.stride(1),
        scale_c.stride(2),
        out.stride(0),
        out.stride(1));
  } else {
    gate_up_silu_tile_inter_scalar_kernel<8, kTileThreads><<<grid, kTileThreads, shared_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr<at::BFloat16>()),
        expert_ids_c.data_ptr<int32_t>(),
        packed_c.data_ptr<uint8_t>(),
        scale_c.data_ptr<float>(),
        global_c.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        packed_c.stride(0),
        packed_c.stride(1),
        packed_c.stride(2),
        scale_c.stride(0),
        scale_c.stride(1),
        scale_c.stride(2),
        out.stride(0),
        out.stride(1));
  }
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "gate_up_silu_tile_inter_scalar_kernel launch failed: ", cudaGetErrorString(err));
  return out;
}

torch::Tensor lynn_native_down_weighted_sum_scalar(
    torch::Tensor inter,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale) {
  TORCH_CHECK(inter.is_cuda(), "inter must be a CUDA tensor");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be a CUDA tensor");
  TORCH_CHECK(routing_weights.is_cuda(), "routing_weights must be a CUDA tensor");
  TORCH_CHECK(down_packed.is_cuda(), "down_packed must be a CUDA tensor");
  TORCH_CHECK(down_scale.is_cuda(), "down_scale must be a CUDA tensor");
  TORCH_CHECK(down_global_scale.is_cuda(), "down_global_scale must be a CUDA tensor");
  TORCH_CHECK(inter.scalar_type() == torch::kBFloat16, "inter must be bfloat16");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(routing_weights.scalar_type() == torch::kFloat32, "routing_weights must be float32");
  TORCH_CHECK(down_packed.scalar_type() == torch::kUInt8, "down_packed must be uint8");
  TORCH_CHECK(down_scale.scalar_type() == torch::kFloat32, "down_scale must be float32");
  TORCH_CHECK(down_global_scale.scalar_type() == torch::kFloat32, "down_global_scale must be float32");
  TORCH_CHECK(inter.dim() == 2 && inter.size(1) == kIntermediate, "inter must be [top_k, 512]");
  TORCH_CHECK(expert_ids.dim() == 1 && expert_ids.size(0) == inter.size(0), "expert_ids must match top_k");
  TORCH_CHECK(routing_weights.dim() == 1 && routing_weights.size(0) == inter.size(0), "routing_weights must match top_k");
  TORCH_CHECK(down_packed.dim() == 3, "down_packed must be [experts, 2048, 256]");
  TORCH_CHECK(down_scale.dim() == 3, "down_scale must be [experts, 2048, 32]");
  TORCH_CHECK(down_packed.size(1) == kHidden, "down_packed row dim must be 2048");
  TORCH_CHECK(down_packed.size(2) == kIntermediate / 2, "down_packed packed inter dim must be 256");
  TORCH_CHECK(down_scale.size(1) == kHidden, "down_scale row dim must be 2048");
  TORCH_CHECK(down_scale.size(2) == kIntermediate / 16, "down_scale group dim must be 32");
  TORCH_CHECK(down_global_scale.numel() == 1, "down_global_scale must be scalar");

  auto inter_c = inter.contiguous();
  auto expert_ids_c = expert_ids.contiguous();
  auto routing_weights_c = routing_weights.contiguous();
  auto packed_c = down_packed.contiguous();
  auto scale_c = down_scale.contiguous();
  auto global_c = down_global_scale.contiguous();
  auto out = torch::empty({kHidden}, inter.options());

  const int64_t top_k = expert_ids_c.numel();
  const size_t shared_bytes = static_cast<size_t>(kThreads) * sizeof(float);
  down_weighted_sum_scalar_kernel<<<kHidden, kThreads, shared_bytes>>>(
      reinterpret_cast<const __nv_bfloat16*>(inter_c.data_ptr<at::BFloat16>()),
      expert_ids_c.data_ptr<int32_t>(),
      routing_weights_c.data_ptr<float>(),
      packed_c.data_ptr<uint8_t>(),
      scale_c.data_ptr<float>(),
      global_c.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      inter_c.stride(0),
      inter_c.stride(1),
      packed_c.stride(0),
      packed_c.stride(1),
      packed_c.stride(2),
      scale_c.stride(0),
      scale_c.stride(1),
      scale_c.stride(2),
      top_k);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "down_weighted_sum_scalar_kernel launch failed: ", cudaGetErrorString(err));
  return out;
}

torch::Tensor lynn_native_down_weighted_sum_tile_scalar(
    torch::Tensor inter,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale,
    int64_t tile_hidden) {
  TORCH_CHECK(inter.is_cuda(), "inter must be a CUDA tensor");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be a CUDA tensor");
  TORCH_CHECK(routing_weights.is_cuda(), "routing_weights must be a CUDA tensor");
  TORCH_CHECK(down_packed.is_cuda(), "down_packed must be a CUDA tensor");
  TORCH_CHECK(down_scale.is_cuda(), "down_scale must be a CUDA tensor");
  TORCH_CHECK(down_global_scale.is_cuda(), "down_global_scale must be a CUDA tensor");
  TORCH_CHECK(inter.scalar_type() == torch::kBFloat16, "inter must be bfloat16");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(routing_weights.scalar_type() == torch::kFloat32, "routing_weights must be float32");
  TORCH_CHECK(down_packed.scalar_type() == torch::kUInt8, "down_packed must be uint8");
  TORCH_CHECK(down_scale.scalar_type() == torch::kFloat32, "down_scale must be float32");
  TORCH_CHECK(down_global_scale.scalar_type() == torch::kFloat32, "down_global_scale must be float32");
  TORCH_CHECK(inter.dim() == 2 && inter.size(1) == kIntermediate, "inter must be [top_k, 512]");
  TORCH_CHECK(expert_ids.dim() == 1 && expert_ids.size(0) == inter.size(0), "expert_ids must match top_k");
  TORCH_CHECK(routing_weights.dim() == 1 && routing_weights.size(0) == inter.size(0), "routing_weights must match top_k");
  TORCH_CHECK(down_packed.dim() == 3, "down_packed must be [experts, 2048, 256]");
  TORCH_CHECK(down_scale.dim() == 3, "down_scale must be [experts, 2048, 32]");
  TORCH_CHECK(down_packed.size(1) == kHidden, "down_packed row dim must be 2048");
  TORCH_CHECK(down_packed.size(2) == kIntermediate / 2, "down_packed packed inter dim must be 256");
  TORCH_CHECK(down_scale.size(1) == kHidden, "down_scale row dim must be 2048");
  TORCH_CHECK(down_scale.size(2) == kIntermediate / 16, "down_scale group dim must be 32");
  TORCH_CHECK(down_global_scale.numel() == 1, "down_global_scale must be scalar");
  TORCH_CHECK(
      tile_hidden == 1 || tile_hidden == 2 || tile_hidden == 4 || tile_hidden == 8,
      "tile_hidden must be one of {1, 2, 4, 8}");

  auto inter_c = inter.contiguous();
  auto expert_ids_c = expert_ids.contiguous();
  auto routing_weights_c = routing_weights.contiguous();
  auto packed_c = down_packed.contiguous();
  auto scale_c = down_scale.contiguous();
  auto global_c = down_global_scale.contiguous();
  auto out = torch::empty({kHidden}, inter.options());

  constexpr int kTileThreads = 128;
  const int64_t top_k = expert_ids_c.numel();
  const unsigned int grid_x = static_cast<unsigned int>((kHidden + tile_hidden - 1) / tile_hidden);
  const size_t shared_bytes = static_cast<size_t>(tile_hidden) * kTileThreads * sizeof(float);

  if (tile_hidden == 1) {
    down_weighted_sum_tile_scalar_kernel<1, kTileThreads><<<grid_x, dim3(kTileThreads, 1), shared_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(inter_c.data_ptr<at::BFloat16>()),
        expert_ids_c.data_ptr<int32_t>(),
        routing_weights_c.data_ptr<float>(),
        packed_c.data_ptr<uint8_t>(),
        scale_c.data_ptr<float>(),
        global_c.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        inter_c.stride(0),
        inter_c.stride(1),
        packed_c.stride(0),
        packed_c.stride(1),
        packed_c.stride(2),
        scale_c.stride(0),
        scale_c.stride(1),
        scale_c.stride(2),
        top_k);
  } else if (tile_hidden == 2) {
    down_weighted_sum_tile_scalar_kernel<2, kTileThreads><<<grid_x, dim3(kTileThreads, 2), shared_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(inter_c.data_ptr<at::BFloat16>()),
        expert_ids_c.data_ptr<int32_t>(),
        routing_weights_c.data_ptr<float>(),
        packed_c.data_ptr<uint8_t>(),
        scale_c.data_ptr<float>(),
        global_c.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        inter_c.stride(0),
        inter_c.stride(1),
        packed_c.stride(0),
        packed_c.stride(1),
        packed_c.stride(2),
        scale_c.stride(0),
        scale_c.stride(1),
        scale_c.stride(2),
        top_k);
  } else if (tile_hidden == 4) {
    down_weighted_sum_tile_scalar_kernel<4, kTileThreads><<<grid_x, dim3(kTileThreads, 4), shared_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(inter_c.data_ptr<at::BFloat16>()),
        expert_ids_c.data_ptr<int32_t>(),
        routing_weights_c.data_ptr<float>(),
        packed_c.data_ptr<uint8_t>(),
        scale_c.data_ptr<float>(),
        global_c.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        inter_c.stride(0),
        inter_c.stride(1),
        packed_c.stride(0),
        packed_c.stride(1),
        packed_c.stride(2),
        scale_c.stride(0),
        scale_c.stride(1),
        scale_c.stride(2),
        top_k);
  } else {
    down_weighted_sum_tile_scalar_kernel<8, kTileThreads><<<grid_x, dim3(kTileThreads, 8), shared_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(inter_c.data_ptr<at::BFloat16>()),
        expert_ids_c.data_ptr<int32_t>(),
        routing_weights_c.data_ptr<float>(),
        packed_c.data_ptr<uint8_t>(),
        scale_c.data_ptr<float>(),
        global_c.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        inter_c.stride(0),
        inter_c.stride(1),
        packed_c.stride(0),
        packed_c.stride(1),
        packed_c.stride(2),
        scale_c.stride(0),
        scale_c.stride(1),
        scale_c.stride(2),
        top_k);
  }
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "down_weighted_sum_tile_scalar_kernel launch failed: ", cudaGetErrorString(err));
  return out;
}

torch::Tensor lynn_native_down_grouped_per16_reference(
    torch::Tensor inter,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale) {
  TORCH_CHECK(inter.is_cuda(), "inter must be a CUDA tensor");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be a CUDA tensor");
  TORCH_CHECK(routing_weights.is_cuda(), "routing_weights must be a CUDA tensor");
  TORCH_CHECK(down_packed.is_cuda(), "down_packed must be a CUDA tensor");
  TORCH_CHECK(down_scale.is_cuda(), "down_scale must be a CUDA tensor");
  TORCH_CHECK(down_global_scale.is_cuda(), "down_global_scale must be a CUDA tensor");
  TORCH_CHECK(inter.scalar_type() == torch::kBFloat16, "inter must be bfloat16");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(routing_weights.scalar_type() == torch::kFloat32, "routing_weights must be float32");
  TORCH_CHECK(down_packed.scalar_type() == torch::kUInt8, "down_packed must be uint8");
  TORCH_CHECK(down_scale.scalar_type() == torch::kFloat32, "down_scale must be float32");
  TORCH_CHECK(down_global_scale.scalar_type() == torch::kFloat32, "down_global_scale must be float32");
  TORCH_CHECK(inter.dim() == 2 && inter.size(1) == kIntermediate, "inter must be [top_k, 512]");
  TORCH_CHECK(expert_ids.dim() == 1 && expert_ids.size(0) == inter.size(0), "expert_ids must match top_k");
  TORCH_CHECK(
      routing_weights.dim() == 1 && routing_weights.size(0) == inter.size(0),
      "routing_weights must match top_k");
  TORCH_CHECK(down_packed.dim() == 3, "down_packed must be [experts, 2048, 256]");
  TORCH_CHECK(down_scale.dim() == 3, "down_scale must be [experts, 2048, 32]");
  TORCH_CHECK(down_packed.size(1) == kHidden, "down_packed row dim must be 2048");
  TORCH_CHECK(down_packed.size(2) == kIntermediate / 2, "down_packed packed inter dim must be 256");
  TORCH_CHECK(down_scale.size(1) == kHidden, "down_scale row dim must be 2048");
  TORCH_CHECK(down_scale.size(2) == kIntermediate / 16, "down_scale group dim must be 32");
  TORCH_CHECK(down_global_scale.numel() == 1, "down_global_scale must be scalar");

  return lynn_native_down_weighted_sum_scalar(
      inter,
      expert_ids,
      routing_weights,
      down_packed,
      down_scale,
      down_global_scale);
}

torch::Tensor lynn_native_down_grouped_per16_tile_reference(
    torch::Tensor inter,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale,
    int64_t tile_hidden) {
  return lynn_native_down_weighted_sum_tile_scalar(
      inter,
      expert_ids,
      routing_weights,
      down_packed,
      down_scale,
      down_global_scale,
      tile_hidden);
}

torch::Tensor lynn_native_active_moe_scalar_contract(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale) {
  auto inter = lynn_native_gate_up_silu_scalar(
      x,
      expert_ids,
      gate_up_packed,
      gate_up_scale,
      gate_up_global_scale);
  return lynn_native_down_weighted_sum_scalar(
      inter,
      expert_ids,
      routing_weights,
      down_packed,
      down_scale,
      down_global_scale);
}

torch::Tensor lynn_native_active_moe_fused_atomic_scalar(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be a CUDA tensor");
  TORCH_CHECK(routing_weights.is_cuda(), "routing_weights must be a CUDA tensor");
  TORCH_CHECK(gate_up_packed.is_cuda(), "gate_up_packed must be a CUDA tensor");
  TORCH_CHECK(gate_up_scale.is_cuda(), "gate_up_scale must be a CUDA tensor");
  TORCH_CHECK(gate_up_global_scale.is_cuda(), "gate_up_global_scale must be a CUDA tensor");
  TORCH_CHECK(down_packed.is_cuda(), "down_packed must be a CUDA tensor");
  TORCH_CHECK(down_scale.is_cuda(), "down_scale must be a CUDA tensor");
  TORCH_CHECK(down_global_scale.is_cuda(), "down_global_scale must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(routing_weights.scalar_type() == torch::kFloat32, "routing_weights must be float32");
  TORCH_CHECK(gate_up_packed.scalar_type() == torch::kUInt8, "gate_up_packed must be uint8");
  TORCH_CHECK(gate_up_scale.scalar_type() == torch::kFloat32, "gate_up_scale must be float32");
  TORCH_CHECK(down_packed.scalar_type() == torch::kUInt8, "down_packed must be uint8");
  TORCH_CHECK(down_scale.scalar_type() == torch::kFloat32, "down_scale must be float32");
  TORCH_CHECK(x.dim() == 1 && x.numel() == kHidden, "x must be [2048]");
  TORCH_CHECK(expert_ids.dim() == 1, "expert_ids must be [top_k]");
  TORCH_CHECK(routing_weights.dim() == 1 && routing_weights.size(0) == expert_ids.size(0),
              "routing_weights must match expert_ids");
  TORCH_CHECK(gate_up_packed.dim() == 3, "gate_up_packed must be [experts, 1024, 1024]");
  TORCH_CHECK(gate_up_scale.dim() == 3, "gate_up_scale must be [experts, 1024, 128]");
  TORCH_CHECK(down_packed.dim() == 3, "down_packed must be [experts, 2048, 256]");
  TORCH_CHECK(down_scale.dim() == 3, "down_scale must be [experts, 2048, 32]");

  auto xc = x.contiguous();
  auto expert_ids_c = expert_ids.contiguous();
  auto routing_weights_c = routing_weights.contiguous();
  auto gate_packed_c = gate_up_packed.contiguous();
  auto gate_scale_c = gate_up_scale.contiguous();
  auto gate_global_c = gate_up_global_scale.contiguous();
  auto down_packed_c = down_packed.contiguous();
  auto down_scale_c = down_scale.contiguous();
  auto down_global_c = down_global_scale.contiguous();
  auto out = torch::zeros({kHidden}, x.options().dtype(torch::kFloat32));

  const int64_t top_k = expert_ids_c.numel();
  const dim3 grid(static_cast<unsigned int>(top_k), kIntermediate);
  const size_t shared_bytes = static_cast<size_t>(kThreads) * 2 * sizeof(float);
  active_moe_fused_atomic_scalar_kernel<<<grid, kThreads, shared_bytes>>>(
      reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr<at::BFloat16>()),
      expert_ids_c.data_ptr<int32_t>(),
      routing_weights_c.data_ptr<float>(),
      gate_packed_c.data_ptr<uint8_t>(),
      gate_scale_c.data_ptr<float>(),
      gate_global_c.data_ptr<float>(),
      down_packed_c.data_ptr<uint8_t>(),
      down_scale_c.data_ptr<float>(),
      down_global_c.data_ptr<float>(),
      out.data_ptr<float>(),
      gate_packed_c.stride(0),
      gate_packed_c.stride(1),
      gate_packed_c.stride(2),
      gate_scale_c.stride(0),
      gate_scale_c.stride(1),
      gate_scale_c.stride(2),
      down_packed_c.stride(0),
      down_packed_c.stride(1),
      down_packed_c.stride(2),
      down_scale_c.stride(0),
      down_scale_c.stride(1),
      down_scale_c.stride(2));
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "active_moe_fused_atomic_scalar_kernel launch failed: ", cudaGetErrorString(err));
  return out;
}

torch::Tensor lynn_native_active_moe_grouped_per16_contract(
    torch::Tensor x,
    torch::Tensor expert_ids,
    torch::Tensor routing_weights,
    torch::Tensor gate_up_packed,
    torch::Tensor gate_up_scale,
    torch::Tensor gate_up_global_scale,
    torch::Tensor down_packed,
    torch::Tensor down_scale,
    torch::Tensor down_global_scale) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be a CUDA tensor");
  TORCH_CHECK(routing_weights.is_cuda(), "routing_weights must be a CUDA tensor");
  TORCH_CHECK(gate_up_packed.is_cuda(), "gate_up_packed must be a CUDA tensor");
  TORCH_CHECK(gate_up_scale.is_cuda(), "gate_up_scale must be a CUDA tensor");
  TORCH_CHECK(gate_up_global_scale.is_cuda(), "gate_up_global_scale must be a CUDA tensor");
  TORCH_CHECK(down_packed.is_cuda(), "down_packed must be a CUDA tensor");
  TORCH_CHECK(down_scale.is_cuda(), "down_scale must be a CUDA tensor");
  TORCH_CHECK(down_global_scale.is_cuda(), "down_global_scale must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(routing_weights.scalar_type() == torch::kFloat32, "routing_weights must be float32");
  TORCH_CHECK(gate_up_packed.scalar_type() == torch::kUInt8, "gate_up_packed must be uint8");
  TORCH_CHECK(gate_up_scale.scalar_type() == torch::kFloat32, "gate_up_scale must be float32");
  TORCH_CHECK(gate_up_global_scale.scalar_type() == torch::kFloat32, "gate_up_global_scale must be float32");
  TORCH_CHECK(down_packed.scalar_type() == torch::kUInt8, "down_packed must be uint8");
  TORCH_CHECK(down_scale.scalar_type() == torch::kFloat32, "down_scale must be float32");
  TORCH_CHECK(down_global_scale.scalar_type() == torch::kFloat32, "down_global_scale must be float32");
  TORCH_CHECK(x.dim() == 1 && x.numel() == kHidden, "x must be [2048]");
  TORCH_CHECK(expert_ids.dim() == 1, "expert_ids must be [top_k]");
  TORCH_CHECK(
      routing_weights.dim() == 1 && routing_weights.size(0) == expert_ids.size(0),
      "routing_weights must match expert_ids");
  TORCH_CHECK(gate_up_packed.dim() == 3, "gate_up_packed must be [experts, 1024, 1024]");
  TORCH_CHECK(gate_up_scale.dim() == 3, "gate_up_scale must be [experts, 1024, 128]");
  TORCH_CHECK(down_packed.dim() == 3, "down_packed must be [experts, 2048, 256]");
  TORCH_CHECK(down_scale.dim() == 3, "down_scale must be [experts, 2048, 32]");
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

  TORCH_CHECK(
      false,
      "active_moe_grouped_per16_contract passed shape/layout checks, but the "
      "grouped per-16 native-FP4 active expert FFN kernel is not implemented "
      "yet. This guarded ABI exists so the future CUTLASS/custom CUDA kernel "
      "can replace only the inner math without changing Python/runtime layout.");
}
