#!/usr/bin/env bash

#SBATCH -p instruction              # Specify the partition
#SBATCH -J Task6Job                 # Job name
#SBATCH -o Task6Output.out          # Standard output file
#SBATCH -e Task6Error.err           # Error output file
#SBATCH -c 1                        # Request 1 CPU core

# Execute the program
./task6 6  # Replace 6 with the desired value of N

