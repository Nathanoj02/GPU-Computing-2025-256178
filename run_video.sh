#!/bin/bash

#SBATCH --job-name=video_processing
#SBATCH --output=cluster/output_%j.out
#SBATCH --error=cluster/error_%j.err
#SBATCH --partition=edu-short
#SBATCH --gres=gpu:a30.24:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# Load required modules
module load NVHPC/nvhpc/24.7
module load OpenCV/4.8.1-foss-2023a-CUDA-12.1.1-contrib

# Compile
make

# Get GPU info
nvidia-smi

# Run executable
./bin/main_video -v dataset/walking_1080.mp4 -k 3
./bin/main_video -v dataset/walking_720.mp4 -k 3
./bin/main_video -v dataset/walking_480.mp4 -k 3
./bin/main_video -v dataset/walking_240.mp4 -k 3
