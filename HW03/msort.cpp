#include <omp.h>
#include <algorithm>  // for std::copy and std::merge
#include "msort.h"     // msort function prototype
#include <iostream>
#include <cstdlib>
// 辅助函数，用于合并两个有序部分
void merge(int* arr, int* temp, int left, int mid, int right) {
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    std::copy(temp + left, temp + right + 1, arr + left);
}

// 插入排序用于小规模数组的排序
void insertion_sort(int* arr, int left, int right) {
    for (int i = left + 1; i <= right; ++i) {
        int key = arr[i];
        int j = i - 1;
        while (j >= left && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// 并行归并排序主函数
void parallel_merge_sort(int* arr, int* temp, int left, int right, int ts) {
    if (right - left <= ts) {
        // 如果数组规模小于阈值，使用插入排序
        insertion_sort(arr, left, right);
        return;
    }

    int mid = (left + right) / 2;

    #pragma omp task shared(arr, temp)
    {
        parallel_merge_sort(arr, temp, left, mid, ts);
    }

    #pragma omp task shared(arr, temp)
    {
        parallel_merge_sort(arr, temp, mid + 1, right, ts);
    }

    #pragma omp taskwait  // 等待两个子任务完成
    merge(arr, temp, left, mid, right);
}

// msort 函数的实现
void msort(int* arr, int n, int ts) {
    int* temp = new int[n];
    #pragma omp parallel
    {
        #pragma omp single  // 创建一个单一的并行任务
        {
            parallel_merge_sort(arr, temp, 0, n - 1, ts);
        }
    }
    delete[] temp;
}

// int main()
// {
//     int arr[] = {5,3,8,2,7,4,1,9,0};
//     msort(arr, 9, 1);
//     for (int i = 0; i < 9; i++)
//     {
//         std::cout << arr[i] << std::endl;
//     }
//     return 0;
// }