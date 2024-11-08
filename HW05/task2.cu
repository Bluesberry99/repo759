#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

__global__ void computeKernel(int *dA, int a) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    int idx = y * blockDim.x + x; // 计算唯一索引

    // 计算 a * x + y 并存储在 dA 中
    dA[idx] = a * x + y;
}

int main() {
    const int arraySize = 16;
    int hA[arraySize]; // 主机数组
    int *dA; // 设备数组指针

    // 生成随机的 a 值
    srand(time(0));
    int a = rand() % 10 + 1; // 随机整数，范围 1 到 10

    // 在设备上分配内存
    cudaMalloc(&dA, arraySize * sizeof(int));

    // 启动核函数，2 个 block，每个 block 8 个线程
    computeKernel<<<2, 8>>>(dA, a);

    // 将数据从设备复制到主机
    cudaMemcpy(hA, dA, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

//    std::cout << "The value of a is: " << a << std::endl;

    // 输出结果，每个值用空格隔开
    for (int i = 0; i < arraySize; i++) {
        std::cout << hA[i] << " ";
    }
    std::cout << std::endl;

    // 释放设备内存
    cudaFree(dA);

    return 0;
}
