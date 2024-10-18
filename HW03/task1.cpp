#include <iostream>
#include <vector>
#include <cstdlib>   // 用于 rand
#include <ctime>     // 用于 time
#include <chrono>    // 用于高精度计时
#include "matmul.h"
#include <omp.h>


int main(int argc, char* argv[]) {
    int N = std::atoi(argv[1]);  // 可以调整为更大的值，比如 1000, 1500 等
    double* A = new double[N * N];
    double* B = new double[N * N];
    double* C = new double[N * N]();
    omp_set_num_threads(std::atoi(argv[2]));
    // 初始化随机数生成器
    std::srand(static_cast<unsigned int>(std::time(0)));

    // 生成随机矩阵 A 和 B
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<double>(std::rand()) / RAND_MAX; // 生成 [0, 1) 之间的随机数
        B[i] = static_cast<double>(std::rand()) / RAND_MAX; // 生成 [0, 1) 之间的随机数
    }

    // 计算 mmul1 的时间
    auto start = std::chrono::high_resolution_clock::now();
    mmul(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // 打印结果
    std::cout << N << std::endl;
    std::cout << duration.count() << std::endl;
    // 打印 C 的最后一个元素以确认正确性
    std::cout << C[N * N - 1] << std::endl;

    // 释放动态分配的内存
    delete[] A;
    delete[] B;
    delete[] C;


    return 0;
}
