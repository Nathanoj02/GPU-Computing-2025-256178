#!/bin/bash

#SBATCH --job-name=es5
#SBATCH --output=es5_out_%j.out
#SBATCH --error=es5_err_%j.err
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

name=$(hostname)
./es5 ${name}