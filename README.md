# 安装步骤

说明: 
本项目是采用unsloth软件进行微调,50系列显卡当前社区的支持还未正式发布,作者安装遇到各种依赖问题,如果当前有官方支持的版本,请使用官方安装方式

triton,vllm,Xformers 采用打补丁编译安装方式

适配环境如下：
- 系统：Ubuntu 22.04 LTS
- 显卡：NVIDIA GeForce RTX 5090d
- 驱动：cuda-toolkit-12-8
- 用户: root
- venv: /root/venv/
- python: python3.10

某些网站可能需要魔法打开
如: https://github.com
如: https://download.pytorch.org
如: https://ftp.gnu.org

## 1.  配置软件源
自带的软件源非常缓慢,替换成阿里云的,这个是Ubuntu 22.04发行版本的源
```
# 备份文件
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak

# 更新 sources.list 文件为阿里云的 Ubuntu 22.04 (Jammy) 镜像源
echo "deb http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse" | sudo tee /etc/apt/sources.list
echo "deb-src http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse" | sudo tee -a /etc/apt/sources.list
echo "deb http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse" | sudo tee -a /etc/apt/sources.list
echo "deb-src http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse" | sudo tee -a /etc/apt/sources.list
echo "deb http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse" | sudo tee -a /etc/apt/sources.list
echo "deb-src http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse" | sudo tee -a /etc/apt/sources.list
echo "deb http://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse" | sudo tee -a /etc/apt/sources.list
echo "deb-src http://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse" | sudo tee -a /etc/apt/sources.list

# 更新软件包列表
sudo apt update
```
## 2. 安装gcc 14.1.0
系统：Ubuntu没有gcc 14,所以要编译安装,先安装系统自带gcc,然后编译安装gcc-14.1.0
```
sudo apt install build-essential libgmp-dev libmpfr-dev libmpc-dev flex bison
wget https://ftp.gnu.org/gnu/gcc/gcc-14.1.0/gcc-14.1.0.tar.xz
tar -xf gcc-14.1.0.tar.xz
cd gcc-14.1.0
./contrib/download_prerequisites
mkdir build && cd build
../configure --prefix=/opt/gcc-14 --enable-languages=c,c++ --disable-multilib
make -j$(nproc)
sudo make install

```
自带的gcc版本替换成14.1.0的
可以执行which gcc查看配置做软连接
```
ln -sf /opt/gcc-14/bin/gcc /usr/bin/gcc
ln -sf /opt/gcc-14/bin/g++ /usr/bin/g++
```
## 3. 安装CUDA 驱动
```
get https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```
## 4. 安装软件
```
bash install-dep.sh
```
