#!/bin/bash

source /root/venv/bin/activate

# 设置必要的环境变量
export LD_LIBRARY_PATH="/opt/gcc-14/lib64/:$LD_LIBRARY_PATH"
export BNB_CUDA_VERSION=128 # <--- Changed to 121
export CUDA_HOME=/usr/local/cuda

# 显示环境信息
echo "环境变量设置完成："
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "BNB_CUDA_VERSION=$BNB_CUDA_VERSION"
echo "CUDA_HOME=$CUDA_HOME"

# 检查NVIDIA GPU状态
echo "检查NVIDIA GPU状态："
nvidia-smi

# 运行Python脚本
echo "运行main.py..."
python3 main.py