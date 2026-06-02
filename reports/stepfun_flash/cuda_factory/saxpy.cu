#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define CHECK(call)                                                           \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            return 1;                                                         \
        }                                                                     \
    } while (0)

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    const int n = 1 << 20;
    const float a = 3.0f;

    float* h_x = new float[n];
    float* h_y = new float[n];
    for (int i = 0; i < n; ++i) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    float *d_x = nullptr, *d_y = nullptr;
    CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK(cudaMalloc(&d_y, n * sizeof(float)));

    CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice));

    const int blockDim = 256;
    const int gridDim  = (n + blockDim - 1) / blockDim;
    saxpy<<<gridDim, blockDim>>>(n, a, d_x, d_y);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (fabsf(h_y[i] - 5.0f) > 1e-5f) {
            fprintf(stderr, "FAIL at index %d: got %f, expected 5.0f\n", i,
                    h_y[i]);
            ok = false;
            break;
        }
    }

    if (ok) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }

    delete[] h_x;
    delete[] h_y;
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));

    return ok ? 0 : 1;
}
