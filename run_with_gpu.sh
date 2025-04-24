#!/bin/bash

# 设置必要的环境变量
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/
export BNB_CUDA_VERSION=128 # <--- Changed to 121
export CUDA_HOME=/usr/local/cuda

# 配置NVIDIA库
ldconfig /usr/lib64-nvidia

# 安装 PyTorch (cu121)
echo "安装 PyTorch (cu128)..."
pip3 install --pre torch torchvision torchaudio  --index-url https://download.pytorch.org/whl/nightly/cu128

# 安装 unsloth (会自动处理 bitsandbytes 和 triton)
echo "强制重新安装 unsloth 及其依赖 (cu128)..."
pip install "unsloth[cu128-new-kernels]" -U --force-reinstall --no-cache-dir

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