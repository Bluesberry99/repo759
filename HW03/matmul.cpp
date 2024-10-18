#include "matmul.h"
#include <vector>
// mmul1: 标准的行优先乘法
void mmul(double* A, double* B, double* C, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
//            C[i * N + j] = 0; // 初始化 C[i][j] 为 0
            for (int k = 0; k < N; ++k) {
                C[i * N + j] += A[i * N + k] * B[k * N + j]; // 行乘列
            }
        }
    }
}
