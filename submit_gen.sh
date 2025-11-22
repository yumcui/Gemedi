#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C "geforce3090|a5000|a5500"
#SBATCH -t 4:00:00
#SBATCH --mem=32G
#SBATCH -J fake-patient-gen
#SBATCH -o slurm-gen-%j.out

source ~/pytorch.venv/bin/activate
python train_generator.py
