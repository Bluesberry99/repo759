#include <iostream>
#include <cstdlib>   // For atoi, srand, rand
#include <ctime>     // For time()
#include <chrono>    // For high-precision timing
#include "conv.h"

// Generate a random image matrix, with elements ranging from -10.0 to 10.0
void generateImage(float* image, int n) {
    for (int i = 0; i < n * n; ++i) {
        image[i] = -10.0f + 20.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

// Generate a random mask matrix, with elements ranging from -1.0 to 1.0
void generateMask(float* mask, int m) {
    for (int i = 0; i < m * m; ++i) {
        mask[i] = -1.0f + 2.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

int main(int argc, char* argv[]) {
    // Check if the correct number of arguments is provided
    if (argc != 3) {
        std::cerr << "Usage: ./task2 <n> <m>" << std::endl;
        return 1;
    }

    // Convert command-line arguments to integers
    int n = std::atoi(argv[1]);
    int m = std::atoi(argv[2]);

    // Check if m is an odd number
    if (m % 2 == 0) {
        std::cerr << "Error: m must be an odd number" << std::endl;
        return 1;
    }

    // Use the current time as the seed for the random number generator
    std::srand(static_cast<unsigned int>(std::time(0)));

    // Allocate memory for the image and mask
    float* image = new float[n * n];
    float* mask = new float[m * m];
    float* output = new float[n * n];

/*  // example check
    float* output = new float[n * n];

    float image[16] = {1, 3, 4, 8, 6, 5, 2, 4, 3, 4, 6, 8, 1, 4, 5, 2};
    float mask[9] = {0, 0, 1, 0, 1, 0, 1, 0, 0};
*/

    // Generate random image and mask
    generateImage(image, n);
    generateMask(mask, m);

    // Record the start time before calling the convolution function
    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform convolution operation
    convolve(image, mask, output, n, m);

    // Record the end time after the convolution function completes
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    // Output the time taken by the convolution function (in milliseconds)
    std::cout << duration.count() << std::endl;

    // Output the first element of the convolution result matrix
    std::cout << output[0] << std::endl;

    // Output the last element of the convolution result matrix
    std::cout << output[n * n - 1] << std::endl;
/*
    // Output the entire convolution result matrix
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

    // Free the allocated memory
    delete[] image;
    delete[] mask;
    delete[] output;

    // Return 0 to indicate successful execution
    return 0;
}
