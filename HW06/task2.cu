#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include "stencil.cuh"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " n R threads_per_block" << std::endl;
        return -1;
    }

    unsigned int n = std::stoi(argv[1]);
    unsigned int R = std::stoi(argv[2]);
    unsigned int threads_per_block = std::stoi(argv[3]);
    unsigned int mask_size = 2 * R + 1;

    // Allocate host memory
    float* h_image = (float*)malloc(n * sizeof(float));
    float* h_mask = (float*)malloc(mask_size * sizeof(float));
    float* h_output = (float*)malloc(n * sizeof(float));

    // Initialize image and mask with random values
    srand(time(0));
    for (unsigned int i = 0; i < n; ++i) {
        h_image[i] = (float(rand()) / RAND_MAX) * 2 - 1;
    }
    for (unsigned int i = 0; i < mask_size; ++i) {
        h_mask[i] = (float(rand()) / RAND_MAX) * 2 - 1;
    }

    // Allocate device memory
    float *d_image, *d_mask, *d_output;
    cudaMalloc(&d_image, n * sizeof(float));
    cudaMalloc(&d_mask, mask_size * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_image, h_image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, mask_size * sizeof(float), cudaMemcpyHostToDevice);

    // Record the start and stop events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    stencil(d_image, d_mask, d_output, n, R, threads_per_block);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the result from device to host
    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the last element of the result and execution time
    std::cout << "Last element: " << h_output[n - 1] << std::endl;
    std::cout << "Time taken: " << milliseconds << " ms" << std::endl;

    // Free host and device memory
    free(h_image);
    free(h_mask);
    free(h_output);
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
