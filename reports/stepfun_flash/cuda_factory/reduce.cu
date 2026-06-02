// reduce.cu — parallel sum reduction (self-checking, no GPU/nvcc here)
// n = 1<<20 elements, all 1.0f → correct sum = 1048576.0

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA error-check macro
// ---------------------------------------------------------------------------
#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Reduction kernel — shared-memory tree reduction within each block,
// then atomicAdd to a single global accumulator.
// ---------------------------------------------------------------------------
__global__ void reduceKernel(const float * __restrict__ d_in,
                             float * __restrict__ d_out,
                             int n)
{
    // --- shared-memory scratchpad ---
    __shared__ float sdata[256];

    // --- global thread index ---
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;

    // --- each thread loads one element (bounds-checked) ---
    float val = 0.0f;
    if (gid < n) {
        val = d_in[gid];
    }
    sdata[tid] = val;
    __syncthreads();

    // --- tree reduction in shared memory ---
    // Use a fixed stride that halves each round; works for any blockDim <= 256.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // --- thread 0 writes the per-block partial sum ---
    if (tid == 0) {
        cudaAtomicAdd(d_out, sdata[0]);
    }
}

// ---------------------------------------------------------------------------
// main()
// ---------------------------------------------------------------------------
int main()
{
    const int n = 1 << 20;           // 1 048 576 elements
    const float expected = (float)n;  // 1 048 576.0f

    const size_t bytes = n * sizeof(float);

    // --- host allocation ---
    float *h_in = (float *)malloc(bytes);
    if (!h_in) {
        fprintf(stderr, "host malloc failed\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < n; ++i) {
        h_in[i] = 1.0f;
    }

    // --- device allocation ---
    float *d_in  = nullptr;
    float *d_out = nullptr;
    CHECK(cudaMalloc(&d_in,  bytes));
    CHECK(cudaMalloc(&d_out, sizeof(float)));

    // --- initialise output accumulator to 0 on device ---
    CHECK(cudaMemset(d_out, 0, sizeof(float)));

    // --- H2D ---
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // --- launch kernel ---
    const int blockSize = 256;
    const int gridSize  = (n + blockSize - 1) / blockSize;
    reduceKernel<<<gridSize, blockSize>>>(d_in, d_out, n);
    CHECK(cudaGetLastError());   // kernel-launch error check
    CHECK(cudaDeviceSynchronize());

    // --- D2H ---
    float result = 0.0f;
    CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    // --- self-check ---
    bool ok = std::fabs(result - expected) <= 1e-4f * expected;

    if (ok) {
        printf("PASS\n");
    } else {
        printf("FAIL got=%f\n", result);
    }

    // --- cleanup ---
    free(h_in);
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
