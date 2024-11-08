// task3.cu
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <random>
#include "vscale.cuh"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " n" << std::endl;
        return 1;
    }

    unsigned int n = atoi(argv[1]);
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *d_a, *d_b;

    // 使用 C++11 随机数生成器生成数组 a 和 b 的随机数
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source());
    std::uniform_real_distribution<float> dista(-10.0, 10.0);
    std::uniform_real_distribution<float> distb(0.0, 1.0);

    for (unsigned int i = 0; i < n; ++i) {
        h_a[i] = dista(generator);  // 生成 -10 到 10 之间的随机数
        h_b[i] = distb(generator);  // 生成 0 到 1 之间的随机数
        // std::cout << h_a[i] * h_b[i] << std::endl;
    }

    // 分配设备内存
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // 设置 CUDA 事件用于测量内核执行时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 配置并启动 vscale 核函数
    int threadsPerBlock = 512;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    vscale<<<blocks, threadsPerBlock>>>(d_a, d_b, n);
    cudaEventRecord(stop);

    // 等待内核完成
    cudaEventSynchronize(stop);

    // 计算并打印内核执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "runtime: ";
    std::cout << milliseconds << std::endl;

    // 将结果从设备复制回主机
    cudaMemcpy(h_b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);

    // std::cout << "Values stored in hA: " << std::endl;
    // for (unsigned int i = 0; i < n; i++)
    //     std::cout << h_b[i] << std::endl;


    // 打印数组 b 的第一个和最后一个元素
    std::cout << h_b[0] << std::endl;
    std::cout << h_b[n - 1] << std::endl;

    // 释放资源
    delete[] h_a;
    delete[] h_b;
    cudaFree(d_a);
    cudaFree(d_b);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}