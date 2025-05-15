#!/bin/bash

#SBATCH --job-name=spmv
#SBATCH --output=spmv%j.out
#SBATCH --error=spmv%j.err
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

./spmv "$@"