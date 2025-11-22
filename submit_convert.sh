#!/bin/bash
#SBATCH -p batch
#SBATCH -t 0:30:00
#SBATCH --mem=64G
#SBATCH -J convert-gguf
#SBATCH -o slurm-convert-%j.out

source ~/pytorch.venv/bin/activate

python llama.cpp/convert_hf_to_gguf.py ./merged_llama3_patient_gen --outfile patient_gen_f16.gguf --outtype f16
