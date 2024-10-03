#include <iostream>
#include <cstdlib>   // atoi, srand, rand
#include <ctime>     // time()
#include <chrono>    //chrono
#include <random>    // random
#include "scan.h"
#include <cmath>

int main(int argc, char* argv[]) {
    // Check the number of inputs
    if (argc != 2) {
        // If not, display usage information and exit
        std::cerr << "Usage: ./task1 n" << std::endl;
        return 1;
    }

    // Convert the first command-line argument to an integer 'n'
    int n = std::atoi(argv[1]);
    if (n <= 0) {
        // If 'n' is not a positive integer, display an error message and exit
        std::cerr << "Error: n should be a positive integer" << std::endl;
        return 1;
    }
/*
    // Use the current time as the seed for the random number generator
    std::srand(static_cast<unsigned int>(std::cos(std::time(0))));

    // Allocate memory for input and output arrays of size 'n'
    float* input = new float[n];
    float* output = new float[n];

    // Generate 'n' random floating-point numbers between -1.0 and 1.0, stored in the 'input' array
    for (int i = 0; i < n; ++i) {
        // Generate a random floating-point number between -1.0 and 1.0
        input[i] = -1.0f + 2.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
*/

    // Use high-precision time as the seed for the random number generator
    std::random_device rd; 
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> dis(-1.0, 1.0); 

    // Allocate memory for input and output arrays of size 'n'
    float* input = new float[n];
    float* output = new float[n];

    // Generate 'n' random floating-point numbers between -1.0 and 1.0, stored in the 'input' array
    for (int i = 0; i < n; ++i) {
        input[i] = dis(gen);
    }

    // Record the start time before calling the scan function
    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform an inclusive scan on the 'input' array, results are stored in the 'output' array
    scan(input, output, n);

    // Record the end time after the scan function finishes
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds as a double value
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    // Output the time taken by the scan function (in milliseconds)
    std::cout << duration.count() << std::endl;

    // Output the first element of the result array
    std::cout << output[0] << std::endl;

    // Output the last element of the result array
    std::cout << output[n - 1] << std::endl;

    // Free the memory allocated for the input and output arrays
    delete[] input;
    delete[] output;

    // Return 0 to indicate the program executed successfully
    return 0;
}
