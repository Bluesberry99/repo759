#include "matmul.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>


template <typename T>
void initialize_matrix(std::vector<T> &matrix, unsigned int n) {
    for (auto &val : matrix)
        val = static_cast<T>(rand() % 100);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n block_dim\n";
        return -1;
    }

    unsigned int n = atoi(argv[1]);
    unsigned int block_dim = atoi(argv[2]);


    std::vector<int> h_A_int(n * n), h_B_int(n * n), h_C_int(n * n);
    std::vector<float> h_A_float(n * n), h_B_float(n * n), h_C_float(n * n);
    std::vector<double> h_A_double(n * n), h_B_double(n * n), h_C_double(n * n);

    initialize_matrix(h_A_int, n);
    initialize_matrix(h_B_int, n);
    initialize_matrix(h_A_float, n);
    initialize_matrix(h_B_float, n);
    initialize_matrix(h_A_double, n);
    initialize_matrix(h_B_double, n);


    int *d_A_int, *d_B_int, *d_C_int;
    float *d_A_float, *d_B_float, *d_C_float;
    double *d_A_double, *d_B_double, *d_C_double;

    cudaMalloc(&d_A_int, n * n * sizeof(int));
    cudaMalloc(&d_B_int, n * n * sizeof(int));
    cudaMalloc(&d_C_int, n * n * sizeof(int));
    cudaMalloc(&d_A_float, n * n * sizeof(float));
    cudaMalloc(&d_B_float, n * n * sizeof(float));
    cudaMalloc(&d_C_float, n * n * sizeof(float));
    cudaMalloc(&d_A_double, n * n * sizeof(double));
    cudaMalloc(&d_B_double, n * n * sizeof(double));
    cudaMalloc(&d_C_double, n * n * sizeof(double));


    cudaMemcpy(d_A_int, h_A_int.data(), n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_int, h_B_int.data(), n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_float, h_A_float.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_float, h_B_float.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_double, h_A_double.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_double, h_B_double.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);


    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    std::cout << "Testing matmul_1 (int):" << std::endl;
    cudaEventRecord(start1);
    matmul_1(d_A_int, d_B_int, d_C_int, n, block_dim);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float time_1 = 0;
    cudaEventElapsedTime(&time_1, start1, stop1);
    cudaMemcpy(h_C_int.data(), d_C_int, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "First element: " << h_C_int[0] << std::endl;
    std::cout << "Last element: " << h_C_int[n * n - 1] << std::endl;
    std::cout << "Time taken: " << time_1 << " ms" << std::endl;
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    //matmul_2
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    std::cout << "Testing matmul_2 (float):" << std::endl;
    cudaEventRecord(start2);
    matmul_2(d_A_float, d_B_float, d_C_float, n, block_dim);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float time_2 = 0;
    cudaEventElapsedTime(&time_2, start2, stop2);
    cudaMemcpy(h_C_float.data(), d_C_float, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "First element: " << h_C_float[0] << std::endl;
    std::cout << "Last element: " << h_C_float[n * n - 1] << std::endl;
    std::cout << "Time taken: " << time_2 << " ms" << std::endl;
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    //matmul_3
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    std::cout << "Testing matmul_3 (double):" << std::endl;
    cudaEventRecord(start3);
    matmul_3(d_A_double, d_B_double, d_C_double, n, block_dim);
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    float time_3 = 0;
    cudaEventElapsedTime(&time_3, start3, stop3);
    cudaMemcpy(h_C_double.data(), d_C_double, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "First element: " << h_C_double[0] << std::endl;
    std::cout << "Last element: " << h_C_double[n * n - 1] << std::endl;
    std::cout << "Time taken: " << time_3 << " ms" << std::endl;
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);


    cudaFree(d_A_int);
    cudaFree(d_B_int);
    cudaFree(d_C_int);
    cudaFree(d_A_float);
    cudaFree(d_B_float);
    cudaFree(d_C_float);
    cudaFree(d_A_double);
    cudaFree(d_B_double);
    cudaFree(d_C_double);

    return 0;
}
