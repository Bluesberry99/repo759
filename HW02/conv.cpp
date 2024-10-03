#include "conv.h"

// conv function
void convolve(const float* image, const float* mask, float* output, int n, int m) {
    int half_m = m / 2;

    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < n; ++y) {
            float sum = 0.0f;

            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    // calculate place in the image
                    int img_x = x + i - half_m;
                    int img_y = y + j - half_m;

                    // check boundage condition
                    float img_value;
                    if (img_x >= 0 && img_x < n && img_y >= 0 && img_y < n) {
                        img_value = image[img_x * n + img_y];
                    } else if ((img_x >= 0 && img_x < n) || (img_y >= 0 && img_y < n)) {
                        img_value = 1.0f; // boundage set 1
                    } else {
                        img_value = 0.0f; // out of boundage set 0
                    }

                    // summary
                    sum += mask[i * m + j] * img_value;
                }
            }
            output[x * n + y] = sum;
        }
    }
}
