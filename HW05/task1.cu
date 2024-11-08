#include <cstdio>
#include <cuda_runtime.h>

__global__ void factorialKernel() {
    int a = threadIdx.x + 1; // Calculate factorial for numbers 1 to 8
    int b = 1;
    for (int i = 1; i <= a; i++) {
        b *= i;
    }
    printf("%d! = %d\n", a, b);
}

int main() {
    factorialKernel<<<1, 8>>>();
    cudaDeviceSynchronize(); // Ensure all threads complete before exiting
    return 0;
}
