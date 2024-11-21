#include <cuda_runtime.h>
#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    // Calculate the global thread ID
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n * n) {
        size_t row = idx / n;
        size_t col = idx % n;

        float sum = 0.0f;
        for (size_t k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    size_t num_elements = n * n;
    size_t num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    matmul_kernel<<<num_blocks, threads_per_block>>>(A, B, C, n);

    // Synchronize the device (optional if using cudaEventSynchronize elsewhere)
    cudaDeviceSynchronize();
}
