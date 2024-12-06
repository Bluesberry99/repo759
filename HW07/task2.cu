#include "reduce.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

// Initialize the array with random values
void initialize_array(std::vector<float> &array, unsigned int N) {
    for (auto &val : array) {
        val = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Range [-1, 1]
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task2 N threads_per_block\n";
        return -1;
    }

    unsigned int N = atoi(argv[1]);
    unsigned int threads_per_block = atoi(argv[2]);

    // Host array
    std::vector<float> h_input(N);
    initialize_array(h_input, N);

    // Device arrays
    float *d_input, *d_output;
    unsigned int blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, blocks * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Call reduce
    cudaEventRecord(start);
    reduce(&d_input, &d_output, N, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result back to host
    float result;
    cudaMemcpy(&result, d_input, sizeof(float), cudaMemcpyDeviceToHost);

    // Measure time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output results
    std::cout << "Sum: " << result << std::endl;
    std::cout << "Time taken: " << milliseconds << " ms" << std::endl;

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
