#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void add_one_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = x[idx] + 1.0f;
  }
}

}  // namespace

torch::Tensor lynn_native_add_one(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  auto xc = x.contiguous();
  auto out = torch::empty_like(xc);
  const int64_t n = xc.numel();
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  add_one_kernel<<<blocks, threads>>>(xc.data_ptr<float>(), out.data_ptr<float>(), n);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "add_one_kernel launch failed: ", cudaGetErrorString(err));
  return out;
}
