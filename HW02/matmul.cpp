#include "matmul.h"
#include <vector>
// mmul1: Standard row-major matrix multiplication
void mmul1(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
//            C[i * N + j] = 0; // Initialize C[i][j] to 0
            for (int k = 0; k < N; ++k) {
                C[i * N + j] += A[i * N + k] * B[k * N + j]; // Row by column multiplication
            }
        }
    }
}

// mmul2: Swap the inner loop
void mmul2(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += A[i * N + k] * B[k * N + j]; // Row by column multiplication
            }
        }
    }
}

// mmul3: Outer loop becomes inner loop
void mmul3(double* A, double* B, double* C, int N) {
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
            for (int i = 0; i < N; ++i) {
                C[i * N + j] += A[i * N + k] * B[k * N + j]; // Row by column multiplication
            }
        }
    }
}

// mmul4: Using std::vector
void mmul4(std::vector<double>& A, std::vector<double>& B, std::vector<double>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
 //           C[i * N + j] = 0; // Initialize C[i][j] to 0
            for (int k = 0; k < N; ++k) {
                C[i * N + j] += A[i * N + k] * B[k * N + j]; // Row by column multiplication
            }
        }
    }
}
