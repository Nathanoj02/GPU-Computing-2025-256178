#!/bin/bash

#SBATCH --job-name=metrics
#SBATCH --output=cluster/output_%j.out
#SBATCH --error=cluster/error_%j.err
#SBATCH --partition=edu-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# Load required modules
module load NVHPC/nvhpc/24.7
module load OpenCV/4.8.1-foss-2023a-CUDA-12.1.1-contrib

# Compile
make

# Run executable
./bin/main_metrics -i dataset/frame0_1080.png