#!/usr/bin/env python3
"""下载 GPT-2 模型文件的脚本"""

import os
import json
import requests
from pathlib import Path

# GPT-2 模型文件下载地址（使用 Hugging Face 镜像）
BASE_URL = "https://huggingface.co/gpt2/resolve/main"
MODEL_BASE = "https://openaipublic.blob.core.windows.net/gpt-2"

def download_file(url, local_path):
    """下载文件到本地"""
    print(f"正在下载: {url}")
    print(f"保存到: {local_path}")
    
    # 创建目录
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # 如果文件已存在，跳过
    if os.path.exists(local_path):
        print(f"文件已存在，跳过: {local_path}")
        return
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✓ 下载完成: {local_path}")
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        raise

def download_gpt2_124m():
    """下载 GPT-2 124M 模型文件"""
    # 设置本地存储路径
    local_base = os.path.expanduser("~/gpt-2-models")
    
    # 从 OpenAI 公开存储下载的文件
    openai_files = {
        # 编码文件
        f"{MODEL_BASE}/encodings/main/encoder.json": f"{local_base}/encodings/main/encoder.json",
        f"{MODEL_BASE}/encodings/main/vocab.bpe": f"{local_base}/encodings/main/vocab.bpe",
        # 模型文件
        f"{MODEL_BASE}/models/124M/hparams.json": f"{local_base}/models/124M/hparams.json",
        f"{MODEL_BASE}/models/124M/model.ckpt.data-00000-of-00001": f"{local_base}/models/124M/model.ckpt.data-00000-of-00001",
        f"{MODEL_BASE}/models/124M/model.ckpt.index": f"{local_base}/models/124M/model.ckpt.index",
        f"{MODEL_BASE}/models/124M/model.ckpt.meta": f"{local_base}/models/124M/model.ckpt.meta",
    }
    
    print("=" * 60)
    print("开始下载 GPT-2 124M 模型文件")
    print("=" * 60)
    
    # 下载所有文件
    for url, local_path in openai_files.items():
        try:
            download_file(url, local_path)
        except Exception as e:
            print(f"警告: 无法从 {url} 下载: {e}")
            # 继续下载其他文件
    
    # 创建 checkpoint 文件（TensorFlow 需要）
    checkpoint_path = f"{local_base}/models/124M/checkpoint"
    if not os.path.exists(checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'w') as f:
            f.write('model_checkpoint_path: "model.ckpt"\n')
            f.write('all_model_checkpoint_paths: "model.ckpt"\n')
        print(f"✓ 创建 checkpoint 文件: {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("下载完成！")
    print(f"文件保存在: {local_base}")
    print("=" * 60)
    print("\n提示: 现在需要修改代码以使用本地路径")
    print(f"设置环境变量: export GPT2_MODEL_PATH={local_base}")
    print("=" * 60)
    
    return local_base

if __name__ == "__main__":
    download_gpt2_124m()


# ============================================================
# 开始下载 GPT-2 124M 模型文件
# ============================================================
# 正在下载: https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json
# 保存到: /Users/zhangda/gpt-2-models/encodings/main/encoder.json
# ✓ 下载完成: /Users/zhangda/gpt-2-models/encodings/main/encoder.json
# 正在下载: https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe
# 保存到: /Users/zhangda/gpt-2-models/encodings/main/vocab.bpe
# ✓ 下载完成: /Users/zhangda/gpt-2-models/encodings/main/vocab.bpe

# ============================================================
# 下载完成！
# 文件保存在: /Users/zhangda/gpt-2-models
# ============================================================