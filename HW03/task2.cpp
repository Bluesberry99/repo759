#include <iostream>
#include <cstdlib>   // 用于atoi, srand, rand
#include <ctime>     // 用于time()
#include <chrono>    // 用于高精度计时
#include <random>    // random
#include "conv.h"
#include <cmath>
#include <omp.h>

int main(int argc, char* argv[]) {
    // 检查是否提供了正确数量的参数
    if (argc != 3) {
        std::cerr << "用法: ./task2 <n> <t>" << std::endl;
        return 1;
    }
    omp_set_num_threads(std::atoi(argv[2]));
    // 将命令行参数转换为整数
    int n = std::atoi(argv[1]);
    int m = 3;

    // 使用高精度时间作为随机数生成器的种子
    std::random_device rd; // 非确定性随机数生成器
    std::mt19937 gen(rd()); // 使用Mersenne Twister 19937生成器
    std::uniform_real_distribution<float> disn(-10, 10); // 生成-1.0到1.0之间的随机浮点数
    std::uniform_real_distribution<float> dism(-1.0, 1.0); // 生成-1.0到1.0之间的随机浮点数

    // 使用当前时间作为随机数生成器的种子
//    std::srand(static_cast<unsigned int>(std::time(0)));

    // 分配图像和掩码的内存

    float* image = new float[n * n];
    float* mask = new float[m * m];
    float* output = new float[n * n];

/*
    float* output = new float[n * n];

    float image[16] = {1, 3, 4, 8, 6, 5, 2, 4, 3, 4, 6, 8, 1, 4, 5, 2};
    float mask[9] = {0, 0, 1, 0, 1, 0, 1, 0, 0};
*/


    // 生成随机图像和掩码
    for (int i = 0; i < n * n; ++i) {
        image[i] = disn(gen);
}
    for (int i = 0; i < m * m; ++i) {
        mask[i] = dism(gen);
    }

    // 在调用卷积函数之前记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();

    // 执行卷积操作
    convolve(image, mask, output, n, m);

    // 在卷积函数完成后记录结束时间
    auto end_time = std::chrono::high_resolution_clock::now();

    // 计算以毫秒为单位的持续时间
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    // 输出卷积函数所用的时间（以毫秒为单位）
    std::cout << duration.count() << std::endl;

    // 输出卷积结果矩阵的第一个元素
    std::cout << output[0] << std::endl;

    // 输出卷积结果矩阵的最后一个元素
    std::cout << output[n * n - 1] << std::endl;
/*
    //输出全部卷积结果矩阵
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << image[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cout << mask[i * m + j] << " ";
        }
        std::cout << std::endl;
    }
    
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << output[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
*/

    // 释放分配的内存

    delete[] image;
    delete[] mask;
    delete[] output;

    // 返回0表示程序成功执行
    return 0;
}
