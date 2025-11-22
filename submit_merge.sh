#!/bin/bash
#SBATCH -p gpu
#SBATCH -C "geforce3090|a5000|a5500"
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH --mem=32G
#SBATCH -J merge-model
#SBATCH -o slurm-merge-%j.out

module purge
module load cuda/12.1.1

export CUDA_HOME="/oscar/rt/9.2/software/0.20-generic/0.20.1/opt/spack/linux-rhel9-x86_64_v3/gcc-11.3.1/cuda-12.1.1-ebglvvqo7uhjvhvff2qlsjtjd54louaf"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

source ~/pytorch.venv/bin/activate
python merge_peft.py
