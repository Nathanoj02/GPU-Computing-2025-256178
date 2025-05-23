#!/bin/bash

#SBATCH --job-name=spmv_cuda
#SBATCH --output=spmv_cuda%j.out
#SBATCH --error=spmv_cuda%j.err
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

./bin/spmv_cuda "$@"