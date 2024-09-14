#include <iostream>
#include <cstdio>  // For the printf function
#include <cstdlib> // For the atoi function

int main(int argc, char* argv[]) {
    // Check command-line arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " N" << std::endl;
        return 1;
    }

    // Convert the command-line argument to an integer N
    int N = std::atoi(argv[1]);

    // Use the printf function to print integers from 0 to N in ascending order
    for (int i = 0; i <= N; ++i) {
        printf("%d", i);
        if (i != N) printf(" ");  // Print a space only between numbers
    }
    printf("\n");

    // Use std::cout to print integers from N to 0 in descending order
    for (int i = N; i >= 0; --i) {
        std::cout << i;
        if (i != 0) std::cout << " ";  // Print a space only between numbers
    }
    std::cout << std::endl;

    return 0;
}
