#include "scan.h"
#include <stdio.h>
void scan(const float* input, float* output, int n) {
    if (n <= 0) 
    {
        printf("n <= 0");
        return;  
    }
    output[0] = input[0];  // first element

    for (int i = 1; i < n; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}
