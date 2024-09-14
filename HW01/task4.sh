#!/usr/bin/env bash

#SBATCH -p instruction              # Modify this line to select the 'instruction' partition
#SBATCH -c 2                        # Request 2 CPU cores
#SBATCH -J FirstSlurm               # Set the job name to FirstSlurm
#SBATCH -o FirstSlurm.out           # Redirect standard output to FirstSlurm.out file
#SBATCH -e FirstSlurm.err           # Redirect error output to FirstSlurm.err file

# Run the command to print the hostname of the compute node
hostname
