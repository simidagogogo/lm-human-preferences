from typing import Dict

import tensorflow as tf

from lm_human_preferences.datasets.books import books_generator
from lm_human_preferences.datasets.cnndm import cnndm_generator
from lm_human_preferences.datasets.tldr import tldr_generator

# 全局映射表(数据集名字->数据集对象)
_registry: Dict[str, "Dataset"] = {}

class Dataset:
    """一个用于构建和管理训练数据的数据集注册和生成类.
    结合了注册表模式(Registry Pattern)、实例配置、TF数据流水线生成等最佳实践
    """
    def __init__(self, name, *, generator=None,):
        """初始化数据集
        :param name: 数据集名称, 每个Dataset都要求有唯一name, 自动写入全局字典_registry
        :param generator: 生成器函数
        """
        global _registry
        assert name not in _registry
        _registry[name] = self
        self.name = name
        self.generator = generator

    def tf_dataset(self, sequence_length, *, mode, encoder=None, seed=0, comm=None, shuffle=True,
        repeat_count=None,  # Defaults to infinite repeat
        start_token=None,   # trims so that it starts right after start token
        end_token=None,     # trims off last end_token
        padding_token=None,
    ):
        """将生成器包装成TF数据输入管道
        
        :param sequence_length: 最终输出token长度
        :param mode: 数据生成模式(如“train”、“test”)
        :param encoder: tokenizer(含encode方法, 把文本转token id)
        :param comm: 并行训练用的通信对象(如Horovod分布式训练, 数据分片shard, 数据并行训练)
        :param shuffle: 是否shuffle
        :param repeat_count: epoch轮数, None代表无限
        :param start_token/end_token/padding_token: 序列起始、终止和padding所需的特殊tokenid
        
        :return: 一个随时待命的数据源. 可以对它进行next(tf_dataset)操作来获取下一条数据
        """
        if padding_token is None:
            padding_token = encoder.padding_token # padding_token: 50259
            
        def _generator():
            """数据生成器
            
            :return: 封装好的数据源
            """
            # 遍历一遍数据集集合
            inner_gen = self.generator(mode, seed=seed, shuffle=shuffle, comm=comm)
            # 处理数据集合中的每条数据: 将文本编码为token_id, 并处理为统一长度sequence_length
            for text in inner_gen:
                tokens = encoder.encode(text)
                print(f"mode: {mode}, text: {text}, tokens: {tokens}, len(tokens): {len(tokens)}")
                """
                text: s o q t v. p q v b h. e o t a v. p m q n h w s h o x g o e. h a n t b. b g a k f g h e v y r i t l. v g x w i y q j e v q h x., 
                tokens: [82, 267, 10662, 256, 410, 13, 279, 10662, 410, 275, 289, 13, 304, 267, 256, 257, 410, 13, 279, 285, 10662, 299, 289, 266, 264, 289, 267, 2124, 308, 267, 304, 13, 289, 257, 299, 256, 275, 13, 275, 308, 257, 479, 277, 308, 289, 304, 410, 331, 374, 1312, 256, 300, 13, 410, 308, 2124, 266, 1312, 331, 10662, 474, 304, 410, 10662, 289, 2124, 13], 
                len(tokens): 67
                """
                # 若设置start_token则只保留start_token之后部分
                if start_token is not None:
                    try:
                        first_index = tokens.index(start_token) + 1
                        if first_index < len(tokens):
                            tokens = tokens[first_index:]
                    except:
                        continue

                # 只保留n个token
                tokens = tokens[:sequence_length]

                # 若设置end_token则只保留end_token之前部分(可能有多个end_token, 但只看最后一个end_token)
                if end_token is not None:
                    try:
                        last_index = len(tokens) - tokens[::-1].index(end_token)
                        tokens = tokens[:last_index]
                    except:
                        continue

                # 不足用padding_token补齐
                if len(tokens) < sequence_length:
                    tokens = tokens + [padding_token] * (sequence_length - len(tokens))

                assert len(tokens) == sequence_length
                yield dict(tokens=tokens) # key: "tokens"

        # 包装为TF Dataset
        tf_dataset = tf.data.Dataset.from_generator(
            _generator,
            output_shapes=dict(tokens=(sequence_length,)),
            output_types=dict(tokens=tf.int32),
        )
        tf_dataset = tf_dataset.repeat(repeat_count)

        # 若处于分布式环境下, 自动分片shard(只保留本节点用于训练的数据, 如Horovod/TPU等环境. 数据并行, 加快训练速度)
        if comm is not None:
            num_shards = comm.Get_size()
            shard_idx = comm.Get_rank()
            if num_shards > 1:
                assert seed is not None
                tf_dataset = tf_dataset.shard(num_shards, shard_idx)
        return tf_dataset


def get_dataset(name) -> Dataset:
    """根据name动态获取已注册Dataset实例. 
    常见用法: 
        # 1.自动被注册到_registry
        dataset1 = Dataset("wiki", generator=wiki_generator)
        dataset2 = Dataset("lm1b", generator=lm1b_generator)
        # 2.直接通过name动态获取d, 即dataset1对象
        d = get_dataset("wiki")
        # 3.生成TF数据管道
        d.tf_dataset(...) 
    :param name: dataset名称
    :return: Dataset实例
    """
    global _registry
    return _registry[name]

# 注册各个dataset对象
CnnDm = Dataset("cnndm", generator=cnndm_generator,)    # CNN/Daily Mail-摘要任务
Tldr = Dataset("tldr", generator=tldr_generator,)       # TL;DR-摘要任务
Books = Dataset("books", generator=books_generator,)    # 积极情感/物理描述-风格化续写
