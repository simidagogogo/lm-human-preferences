import os
# 设置 Hugging Face 镜像地址
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import random
import re
import sys
import ftfy
from datasets import load_dataset

def tldr_generator_new(mode, seed=0, shuffle=False, comm=None):
    print(f"Loading dataset for mode: {mode}...")
    
    # 这里直接加载全量，CarperAI/openai_summarize_tldr 不算特别大
    try:
        ds = load_dataset("CarperAI/openai_summarize_tldr")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if mode == 'test':
        split = 'valid'
    else:
        split = mode # 'train' or 'valid'

    # 检查 split 是否存在
    if split not in ds:
        print(f"Error: Split '{split}' not found in dataset keys: {ds.keys()}")
        return

    datas = ds[split]
    
    # 为了测试不等待太久，我们只取前 100 条数据来 shuffle 测试
    # 正式使用时请去掉 [:100]
    datas = datas.select(range(100)) 
    if shuffle:
        print("Shuffling data...")
        # Hugging Face Dataset 也可以转成 list 操作，或者使用 .shuffle()
        # 这里沿用你的逻辑，先转 list 索引
        indices = list(range(len(datas)))
        random.seed(seed)
        random.shuffle(indices)
        
        # 注意：直接用列表推导式遍历 Dataset 对象可能会比较慢
        # 在测试时，我们可以只 yield 乱序后的前几个
        def shuffled_iterator():
            for i in indices:
                yield datas[i]
        data_iter = shuffled_iterator()
    else:
        data_iter = datas

    print("Generator ready. Starting to yield texts...")
    for data in data_iter:
        # Hugging Face 的数据结构通常是 {'prompt': ..., 'label': ...}
        prompt = data['prompt']
        label = data['label']
        
        # 拼接还原为原始格式
        text = prompt + label 
        text = ftfy.fix_text(text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        yield text

if __name__ == "__main__":
    print("--- Testing 'valid' mode ---")
    gen = tldr_generator_new(mode='valid', seed=42, shuffle=False)
    
    try:
        # 打印前 3 条数据看看样子
        for i in range(3):
            text = next(gen)
            print(f"\n[Sample {i+1}]:")
            print("="*40)
            # 只打印前 200 个字符预览，防止刷屏
            # print(text[:200] + "..." if len(text) > 200 else text) 
            print(text)
            print("="*40)
            
            # 检查关键特征：是否包含 TL;DR
            if "TL;DR" in text or "tl;dr" in text:
                print("✅ Found 'TL;DR' tag.")
            else:
                print("⚠️ Warning: 'TL;DR' tag not found in preview (might be further down).")

    except StopIteration:
        print("Dataset is empty!")
    except Exception as e:
        print(f"An error occurred: {e}")

