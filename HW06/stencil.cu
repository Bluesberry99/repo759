#include "stencil.cuh"
#include <cuda_runtime.h>
#include <iostream>

// Kernel: Computes the convolution of `image` and `mask`, storing the result in `output`
__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    extern __shared__ float shared_mem[];
    float* shared_image = shared_mem;                            // Shared memory for image
    float* shared_mask = shared_image + blockDim.x + 2 * R;      // Shared memory for mask
    float* shared_output = shared_mask + (2 * R + 1);
    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + tid;

    // Load mask into shared memory
    if (tid < (2 * R + 1)) {
        shared_mask[tid] = mask[tid];
    }

    // Load image into shared memory
    if (global_id < n) {
        shared_image[tid + R] = image[global_id];
        //printf("Image[%d] = %f\n", global_id, shared_image[threadIdx.x + R]);
    } else {
        shared_image[tid + R] = 1;  // Out-of-bound elements treated as 1
    }

    // Handle halo region for left boundary
    if (tid < R) {
        if (global_id >= R) {
            shared_image[tid] = image[global_id - R];
        } else {
            shared_image[tid] = 1;
        }
        //printf("Left Halo[%d] = %f\n", threadIdx.x, shared_image[threadIdx.x]);
    }

    // Handle halo region for right boundary
    if (tid >= blockDim.x - R) {
        unsigned int right_idx = global_id + R;
        if (right_idx < n) {
            shared_image[tid + 2 * R] = image[right_idx];
        } else {
            shared_image[tid + 2 * R] = 1;
        }
        //printf("Right Halo[%d] = %f\n", threadIdx.x + R + blockDim.x, shared_image[threadIdx.x + R + blockDim.x]);

    }

    __syncthreads(); // Synchronize to ensure all shared memory is loaded

//     // Compute the convolution
//     if (global_id < n) {
//         float result = 0.0f;
//         for (int j = -R; j <= R; ++j) {
//             result += shared_image[tid + R + j] * shared_mask[j + R];
//         }
//         output[global_id] = result;
//          printf("Thread %d: Computed result = %f\n", threadIdx.x, result);
//     }
// }

    if (global_id < n)
    {
        // printf("bIdx = %d, tIdx = %d, =====Mask[%d] = %f\n", blockIdx.x, threadIdx.x, threadIdx.x, shared_mask[threadIdx.x]);
        float sum = 0;
        for (int j = -static_cast<int>(R); j <= static_cast<int>(R); j++)
        {
             //printf("Thread %d: shared_image[%d] = %f, shared_mask[%d] = %f\n", 
                //threadIdx.x, threadIdx.x + R + j, shared_image[threadIdx.x + R + j], j + R, shared_mask[j + R]);
            sum += shared_image[threadIdx.x + R + j] * shared_mask[j + R];
        }
        shared_output[threadIdx.x] = sum;
        //printf("Thread %d: Computed sum = %f\n", threadIdx.x, sum);
    }
    __syncthreads();
    if (global_id < n)
        output[global_id] = shared_output[threadIdx.x];
}
// Host function: Invokes the stencil kernel
void stencil(const float* image,
             const float* mask,
             float* output,
             unsigned int n,
             unsigned int R,
             unsigned int threads_per_block) 
{
    unsigned int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = (threads_per_block + 2 * R) * sizeof(float) + (2 * R + 1) * sizeof(float) + threads_per_block * sizeof(float);

    // Launch the kernel
    stencil_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(image, mask, output, n, R);

    // Ensure the kernel has finished
    cudaDeviceSynchronize();
}
