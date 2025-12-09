#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C "geforce3090|a5000|a5500"
#SBATCH -t 8:00:00
#SBATCH --mem=32G
#SBATCH -o slurm-discriminator-%j.out

module load python/3.11.0s-ixrhc3q cuda/12.1.1

current_directory=$(pwd)
source $current_directory/llama_env/bin/activate

python finetune_discriminator.py \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train_data train_discriminator.jsonl \
  --eval_data eval_discriminator.jsonl \
  --output_dir llama3-discriminator \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --max_length 4096