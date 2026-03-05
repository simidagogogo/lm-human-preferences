import json
import random
import re
import ftfy
from datasets import load_dataset

from lm_human_preferences.utils import gcs


def tldr_generator(mode, seed=0, shuffle=False, comm=None):
    """负责提供高质量、清洗过后的TL;DR(文本摘要配对)数据
    
    在TL;DR数据集中, text通常包含两部分:
        原文(Post): 一段较长的论坛帖子内容
        摘要标识符: 通常是TL;DR:
        摘要(Summary): 作者自己写的总结
    典型格式示例:
        "My girlfriend and I have been dating for 3 years. Recently she started acting weird... [几百字的详细描述] ...so I don't know what to do.
        TL;DR: My girlfriend is acting strange and I need advice on how to talk to her."
    训练时的用法:
        模型会读入整段text. 通常会把 TL;DR: 之前的部分作为Prompt. 把TL;DR: 之后的部分作为Target(Label)
        训练目标是让模型学会: 给定前面的长文, 自动生成后面的摘要
    """
    random.seed(seed)

    if mode == 'test':
        mode = 'valid' # validation set serves as training set, since we don't have access..
    assert mode in ['train', 'valid']

    # 访问地址
    # CarperAI/openai_summarize_tldr: https://huggingface.co/datasets/CarperAI/openai_summarize_tldr/tree/main
    # reddit_tldr: https://huggingface.co/datasets/reddit_tldr/tree/main
    with open(gcs.download_file_cached(
        f'https://openaipublic.blob.core.windows.net/lm-human-preferences/tldr/{mode}-subset.json', 
        comm=comm
    )) as f:
        datas = json.load(f)

    if shuffle:
        random.seed(seed)
        random.shuffle(datas)

    for data in datas:
        text = data['content']
        text = ftfy.fix_text(text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        yield text


def tldr_generator_new(mode, seed=0, shuffle=False, comm=None):
    random.seed(seed)
    # Hugging Face 上的对应数据集名称
    # 'reddit_tldr' 是原始数据集
    # 'openai/summarize_from_feedback' 是 OpenAI 用于RLHF带人类偏好版本
    # 这里我们用最接近原始用途的 'yechen/reddit-tldr' 或 'CarperAI/openai_summarize_tldr'
    
    # 这一版是专门整理好的 TL;DR 数据
    ds = load_dataset("CarperAI/openai_summarize_tldr")
    
    if mode == 'test':
        split = 'valid'
    else:
        split = mode # 'train' or 'valid'

    datas = ds[split]

    if shuffle:
        # Hugging Face Dataset 自带 shuffle 方法，但转成 list 后用 random 也可以
        indices = list(range(len(datas)))
        random.seed(seed)
        random.shuffle(indices)
        datas = [datas[i] for i in indices]

    for data in datas:
        # Hugging Face 的数据结构通常是 {'prompt': ..., 'label': ...}
        # 原始代码需要的是拼在一起的 'content'
        # 格式通常是: 文章内容 + "\nTL;DR: " + 摘要
        prompt = data['prompt']
        label = data['label']
        
        # 拼接还原为原始格式
        text = prompt + label 
        
        text = ftfy.fix_text(text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        yield text