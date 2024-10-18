#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include "msort.h"
#include <random>    // random

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " n t ts" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);  // 数组长度
    int t = std::atoi(argv[2]);  // 使用的线程数量
    int ts = std::atoi(argv[3]); // 任务调度的阈值

    int* arr = new int[n];

    // 生成 [-1000, 1000] 范围内的随机数
    std::random_device rd; // 非确定性随机数生成器
    std::mt19937 gen(rd()); // 使用Mersenne Twister 19937生成器
    std::uniform_real_distribution<float> dis(-1000, 1000); // 生成-1.0到1.0之间的随机浮点数


    for (int i = 0; i < n; ++i) {
        arr[i] = dis(gen);
    }

    omp_set_num_threads(t);  // 设置线程数

    // 开始计时
    double start_time = omp_get_wtime();

    // 调用并行归并排序
    msort(arr, n, ts);

    // 结束计时
    double end_time = omp_get_wtime();

    // 输出结果
    std::cout << n << std::endl; 
    std::cout << t << std::endl; 
    std::cout << ts << std::endl; 
    std::cout << arr[0] << std::endl;       // 输出排序后数组的第一个元素
    std::cout << arr[n - 1] << std::endl;   // 输出排序后数组的最后一个元素
    std::cout << (end_time - start_time) * 1000 << " ms" << std::endl; // 输出排序时间

    delete[] arr;
    return 0;
}
