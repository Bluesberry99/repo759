#include "reduce.cuh"
#include <cuda_runtime.h>
#include <iostream>

// Parallel reduction kernel (Kernel 4: First add during global load)
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float shared_data[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // First add during global load
    shared_data[tid] = (idx < n ? g_idata[idx] : 0.0f) + (idx + blockDim.x < n ? g_idata[idx + blockDim.x] : 0.0f);
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
//            shared_data[tid] += shared_data[tid + s];
            shared_data[tid] += shared_data[tid + s / 2];
            if (s % 2 == 1 && tid == s / 2 - 1)
            {
                shared_data[tid] += shared_data[tid + s / 2 + 1];
            }

        }
        __syncthreads();
    }

    // Write the result of this block to g_odata
    if (tid == 0) {
        g_odata[blockIdx.x] = shared_data[0];
    }
}

// Host function for reduction
__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
    unsigned int blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    unsigned int shared_mem_size = threads_per_block * sizeof(float);

    // Call the reduction kernel repeatedly if needed
    while (N > 1) {
        reduce_kernel<<<blocks, threads_per_block, shared_mem_size>>>(*input, *output, N);
        cudaDeviceSynchronize();

        // Update N and pointers
        N = blocks;
        blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
        float *temp = *input;
        *input = *output;
        *output = temp;
    }
}
