#include <iostream>
#include <vector>
#include <cstdlib>   // For rand
#include <ctime>     // For time
#include <chrono>    // For high-precision timing
#include "matmul.h"

int main() {
    const int N = 1024;  // Can be adjusted to larger values, such as 1000, 1500, etc.
    double* A = new double[N * N];
    double* B = new double[N * N];
    double* C1 = new double[N * N];
    double* C2 = new double[N * N];
    double* C3 = new double[N * N];

    // Initialize the random number generator
    std::srand(static_cast<unsigned int>(std::time(0)));

    // Generate random matrices A and B
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<double>(std::rand()) / RAND_MAX; // Generate random numbers between [0, 1)
        B[i] = static_cast<double>(std::rand()) / RAND_MAX; // Generate random numbers between [0, 1)
    }

    // Measure the time for mmul1
    auto start1 = std::chrono::high_resolution_clock::now();
    mmul1(A, B, C1, N);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration1 = end1 - start1;

    // Measure the time for mmul2
    auto start2 = std::chrono::high_resolution_clock::now();
    mmul2(A, B, C2, N);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration2 = end2 - start2;

    // Measure the time for mmul3
    auto start3 = std::chrono::high_resolution_clock::now();
    mmul3(A, B, C3, N);
    auto end3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration3 = end3 - start3;

    // Print results
    std::cout << N << std::endl;
    std::cout << duration1.count() << std::endl;
    std::cout << duration2.count() << std::endl;
    std::cout << duration3.count() << std::endl;

    // Print the last element of C to confirm correctness
    std::cout << C1[N * N - 1] << std::endl;
    std::cout << C2[N * N - 1] << std::endl;
    std::cout << C3[N * N - 1] << std::endl;

    // Release dynamically allocated memory
    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;
    delete[] C3;

    return 0;
}
