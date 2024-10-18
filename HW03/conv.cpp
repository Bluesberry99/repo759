#include "conv.h"

// 卷积函数实现
void convolve(const float* image, const float* mask, float* output, int n, int m) {
    int half_m = m / 2;
    #pragma omp parallel for collapse(2)
    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < n; ++y) {
            float sum = 0.0f;

            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    // 计算图像中的索引位置
                    int img_x = x + i - half_m;
                    int img_y = y + j - half_m;

                    // 检查边界条件
                    float img_value;
                    if (img_x >= 0 && img_x < n && img_y >= 0 && img_y < n) {
                        img_value = image[img_x * n + img_y];
                    } else if ((img_x >= 0 && img_x < n) || (img_y >= 0 && img_y < n)) {
                        img_value = 1.0f; // 边界处理，边缘填充1
                    } else {
                        img_value = 0.0f; // 边界外填充0
                    }

                    // 累加卷积结果
                    sum += mask[i * m + j] * img_value;
                }
            }
            output[x * n + y] = sum;
        }
    }
}
