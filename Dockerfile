# 核心关键点：必须强制使用 linux/amd64 架构
# 因为 TensorFlow 1.13 没有 ARM64 (Apple Silicon) 的安装包
# 注意：platform 在构建时通过 --platform 参数指定
FROM python:3.7-buster

# 设置工作目录
WORKDIR /app

# 1. 安装系统级依赖
# OpenAI 的旧代码库通常需要 MPI (libopenmpi-dev) 和 cmake
# zlib1g-dev 和 build-essential 用于编译旧版 python 包
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    zlib1g-dev \
    libopenmpi-dev \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 2. 升级 pip
# 旧版 Python 镜像自带的 pip 可能过老，无法解析现代 wheel
RUN pip install --upgrade pip

# 3. 安装核心依赖 (严格锁定版本)
# - tensorflow 1.13.1 (用户指定)
# - numpy 1.16.4 (关键！TF 1.13 与 numpy 1.17+ 不兼容，会报错 "Object arrays cannot be loaded...")
# - mpi4py (OpenAI 基线代码常用)
RUN pip install \
    tensorflow==1.13.1 \
    numpy==1.16.4 \
    mpi4py \
    cloudpickle==1.2.1

# 4. (可选) 安装 OpenAI lm-human-preferences 可能需要的其他工具
# 许多旧版 OpenAI 项目使用 pipenv，这里预装一下
RUN pip install pipenv

# 5. 设置环境变量
# 防止 python 生成 .pyc 文件
ENV PYTHONDONTWRITEBYTECODE=1
# 设置 python 输出不被缓存
ENV PYTHONUNBUFFERED=1

# 默认命令
CMD ["/bin/bash"]