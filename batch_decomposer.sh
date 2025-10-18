#!/bin/bash
#SBATCH --job-name=rin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%A_%a.out

# Load CUDA and Python modules (adjust versions as needed)
# module load cuda/11.8
# module load miniconda3

# Run your Python script
/home/jin.ron/envs/cv_cuda/bin/python main.py 
