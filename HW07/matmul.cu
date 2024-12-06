#include "matmul.cuh"
#include <cuda_runtime.h>
#include <iostream>

// Tiled matrix multiplication kernel
template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n, unsigned int block_dim) {
//    extern __shared__ T shared_tile[]; 
    extern __shared__ unsigned char shared_tile[];
    // T *tileA = shared_tile;           
    // T *tileB = shared_tile + block_dim * block_dim; //
    T *tileA = reinterpret_cast<T*>(shared_tile);                          // Shared memory for A
    T *tileB = reinterpret_cast<T*> (&shared_tile[blockDim.x * blockDim.y * sizeof(T)]); // Shared memory for B

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int row = blockIdx.y * block_dim + ty;
    unsigned int col = blockIdx.x * block_dim + tx;

    T value = 0;

    for (unsigned int i = 0; i < (n + block_dim - 1) / block_dim; ++i) {
        if (row < n && i * block_dim + tx < n)
            tileA[ty * block_dim + tx] = A[row * n + i * block_dim + tx];
        else
            tileA[ty * block_dim + tx] = 0;

        if (col < n && i * block_dim + ty < n)
            tileB[ty * block_dim + tx] = B[(i * block_dim + ty) * n + col];
        else
            tileB[ty * block_dim + tx] = 0;

        __syncthreads();

        for (unsigned int k = 0; k < block_dim; ++k)
            value += tileA[ty * block_dim + k] * tileB[k * block_dim + tx];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = value;
}

// Tiled Matrix Multiplication host functions
template <typename T>
void matmul_template(const T *A, const T *B, T *C, unsigned int n, unsigned int block_dim) {
    dim3 threads(block_dim, block_dim);
    dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(T);

    matmul_kernel<<<grid, threads, shared_mem_size>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    matmul_template(A, B, C, n, block_dim);
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
    matmul_template(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
    matmul_template(A, B, C, n, block_dim);
}
