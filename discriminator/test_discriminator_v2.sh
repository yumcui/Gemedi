#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C "geforce3090|a5000|a5500"
#SBATCH -t 4:00:00
#SBATCH --mem=32G
#SBATCH -o slurm-discriminator-%j.out


module load python/3.11.0s-ixrhc3q cuda/12.1.1

current_directory=$(pwd)
source $current_directory/llama_env/bin/activate

python discriminator_v2.py
