import json
import random
import os
import random as rnd
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 设置HuggingFace镜像地址
from datasets import load_dataset

from lm_human_preferences.utils import gcs


# def books_generator_old(mode, seed=0, shuffle=False, comm=None):
#     """积极情感/物理描述-风格化续写
#     :param mode: train/valid/test
#     :param seed: 
#     :param shuffle:
#     :param comm: 
#     """
#     try:
#         file_path = gcs.download_file_cached(
#             f'https://openaipublic.blob.core.windows.net/lm-human-preferences/datasets/book_passages/{mode}.jsonl', 
#             comm=comm
#         )
#         # TODO: 为什么会进到这里?
#         if os.path.getsize(file_path) == 0 or os.path.getsize(file_path) < 10:
#             print(f'WARNING: Dataset file appears to be empty or placeholder. Using test data generator.')
#             rnd.seed(seed)
#             """
#             随机生成句子, 其中
#               - 单词长度: 1, 小写字母
#               - 每个句子包含单词个数: 5到15
#               - 句子个数: 3到8
#             例如: 
#               c y n i d b m r r. k b f e k r k w h f. n y c t m.
#               u q s z k a m y u q n v r u. t p q v n l q b y u f x c p v. f k k r m c x o m. y k a g s c u l a l h m r q.
#             """
#             while True:
#                 text = '. '.join([
#                     ' '.join([chr(ord('a') + rnd.randint(0, 25)) for _ in range(rnd.randint(5, 15))])            
#                     for _ in range(rnd.randint(3, 8))
#                 ]) + '.'
#                 yield text
#             return
        
#         with open(file_path) as f:
#             datas = [json.loads(line) for line in f if line.strip()]
    
#     except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError) as e:
#         print(f'WARNING: Could not load books dataset: {e}. Using test data generator.')
#         rnd.seed(seed)
#         while True:
#             text = '. '.join([
#                 ' '.join([chr(ord('a') + rnd.randint(0, 25)) for _ in range(rnd.randint(5, 15))]) 
#                 for _ in range(rnd.randint(3, 8))
#             ]) + '.'
#             yield text
#         return
    
#     if shuffle:
#         random.seed(seed)
#         random.shuffle(datas)
    
#     for x in datas:
#         if isinstance(x, dict):
#             # 优先找text字段，没有找content, 再没有就把整个字典转字符串
#             text = x.get('text', x.get('content', str(x)))
#             yield text
#         else:
#             yield str(x)


"""
https://openaipublic.blob.core.windows.net/...这个OpenAI老存储桶链接已经彻底失效.
这段代码所需的 "Book Passages" 数据集，实际上是 BookCorpus 数据集的一个子集或变体。
要修复这段代码，最简单、最标准的替代方案是使用 Hugging Face 的 datasets 库来加载 BookCorpus。
解决方案：使用Hugging Face的BookCorpus
BookCorpus 是一个包含大量未出版书籍文本的大型数据集，广泛用于训练 GPT 系列模型（包括 BERT, GPT-2 等）。它完全可以替代原代码中的数据。
"""

def books_generator(mode, seed=0, shuffle=False, comm=None):
    """使用Hugging Face的BookCorpus数据集替代失效的OpenAI链接
    
    :param mode: BookCorpus的split只有train, valid/test需要手动切分
    :param seed:
    :param shuffle:
    :param comm:
    :return: 
    """
    random.seed(seed)
    
    hf_split = mode
    print(f"Loading BookCorpus dataset from Hugging Face (split={hf_split})...")
    
    # streaming=True 允许流式读取，不用一次性下载 6GB 数据，非常适合生成器
    # ds = load_dataset("bookcorpus", split=hf_split, streaming=True, trust_remote_code=True)
    
    # OpenWebText是网络爬取的长文本，风格与BookCorpus高度一致, 是目前公认的最佳替代
    ds = load_dataset("Skylion007/openwebtext", split=hf_split, streaming=True)

    if mode == 'valid':
        ds = ds.skip(100000) # 跳过前10万条作为验证集
    elif mode == 'test':
        ds = ds.skip(110000) # 跳过更多

    if shuffle:
        ds = ds.shuffle(seed=seed, buffer_size=10000)

    for data in ds:
        text = data.get('text', '')
        if text.strip():
             yield text
