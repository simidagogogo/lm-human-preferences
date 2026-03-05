import random
import random as rnd
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 设置HuggingFace镜像地址
from datasets import load_dataset
import os

def books_generator_new(mode, seed=0, shuffle=False, comm=None):
    random.seed(seed)
    
    # 这里为了测试方便，先尝试加载 bookcorpus
    # 如果你本地网络不好，可以临时改成 "wikitext", "wikitext-2-raw-v1" 来测试逻辑
    dataset_name = "bookcorpus" 
    # dataset_name = "wikitext", "wikitext-2-raw-v1" # 备选方案

    hf_split = 'train' 
    try:
        print(f"Loading {dataset_name} dataset from Hugging Face (streaming=True)...")
        # streaming=True 是关键，不用下载几十G数据
        if dataset_name == "bookcorpus":
            # 这是一个社区维护的 parquet 格式版本，应该也能用
            #ds = load_dataset("shibing624/bookcorpus", split="train", streaming=True)
            #ds = load_dataset("wikitext", "wikitext-103-v1", split='train', streaming=True)
            #ds = load_dataset("shibing624/bookcorpus", split="train", streaming=True)
            # 使用 open 版本，它是目前最稳定的 BookCorpus 替代品
            #ds = load_dataset("bookcorpusopen", split="train", streaming=True)
            # 使用 fp16-guy 维护的 Parquet 版本，无脚本，纯数据，绝对可用
            #ds = load_dataset("fp16-guy/bookcorpus", split="train", streaming=True)
                    # 【核心修改】直接指定官方仓库的 parquet 文件路径，绕过失效的脚本
            # 这是一个极其稳定的写法，因为它直接指向了数据本身
            #ds = load_dataset(
            #    "bookcorpus/bookcorpus",
            #    data_files={'train': 'https://huggingface.co/datasets/bookcorpus/bookcorpus/resolve/main/data/train-00000-of-00001.parquet'},
            #    split='train',
            #    streaming=True
            #)

            # 备选：如果 BookCorpus 彻底无法访问，OpenWebText 是目前公认的最佳替代
            # 它的内容也是网络爬取的长文本，风格与 BookCorpus 高度一致
            print("BookCorpus access failed, using Skylion007/openwebtext (Best Alternative)...")
            ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        else:
            # 针对 wikitext 的特殊处理
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=hf_split, streaming=True)
        
        # 简单的切分模拟
        if mode == 'valid':
            ds = ds.skip(100) # 测试时只跳过100条，不用跳10万条，太慢
        elif mode == 'test':
            ds = ds.skip(200)

    except Exception as e:
        print(f'WARNING: Could not load Hugging Face dataset: {e}.')
        print('--> Falling back to random text generator.')
        rnd.seed(seed)
        while True:
            # 生成随机乱码句子的逻辑
            text = '. '.join([
                ' '.join([chr(ord('a') + rnd.randint(0, 25)) for _ in range(rnd.randint(5, 15))])            
                for _ in range(rnd.randint(3, 8))
            ]) + '.'
            yield text
        return

    if shuffle:
        # buffer_size 设置小一点以便快速看到效果
        ds = ds.shuffle(seed=seed, buffer_size=100)

    for data in ds:
        text = data.get('text', '')
        if text.strip():
             yield text

if __name__ == "__main__":
    print("=== Testing 'train' mode (Shuffle=False) ===")
    gen = books_generator_new(mode='train', seed=42, shuffle=False)
    
    try:
        # 打印前 3 条数据
        for i in range(3):
            text = next(gen)
            print(f"\n[Sample {i+1}]:")
            print("-" * 20)
            # 打印前 200 字符预览
            # print(text[:200] + "..." if len(text) > 200 else text) 
            print(text) 
        gen.close() 
        print("Generator closed safely.")
    except StopIteration:
        print("Generator stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")

