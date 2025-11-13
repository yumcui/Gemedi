#!/bin/bash

# --- 1. SBATCH 指令：(保持不变) ---
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH -t 1:00:00
#SBATCH --mem=32G
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# --- 2. 我们的环境设置：(已修复) ---

echo "作业开始：在 $(hostname) 上运行"

# (!! 关键修复 !!) 
# 加载系统环境，让 sbatch "认识" module 命令
source /etc/profile
echo "--- (1/7) 系统环境加载 ---"

# (关键) 彻底清理环境
module purge
rm -rf pytorch.venv # 删除任何旧的、可能损坏的 venv
echo "--- (2/7) 清理完成 ---"

# (关键) 先加载 CUDA 12.1 
module load cuda/12.1.1
echo "--- (3/7) 加载 CUDA 12.1 成功 ---"

# 创建并激活 venv
python -m venv pytorch.venv
source pytorch.venv/bin/activate
echo "--- (4/7) Venv 创建并激活 ---"

# 升级 pip
pip install --upgrade pip
echo "--- (5/7) Pip 升级完成 ---"

# (关键) 安装匹配 CUDA 12.1 的 PyTorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
echo "--- (6/7) PyTorch (cu121) 安装完成 ---"

# (关键) 安装我们那套“已知兼容”的库
pip install transformers==4.36.2 peft==0.7.1 accelerate==0.25.0 datasets==2.15.0 bitsandbytes==0.41.3 trl==0.7.4 scipy
echo "--- (7/7) Transformers/Peft/TRL/BitsandBytes 安装完成 ---"

# --- 3. 运行我们的 Python 脚本 ---

echo "--- (最终) 开始运行 Python 微调脚本 ---"
python train_lora.py

echo "--- 作业完成 ---"
