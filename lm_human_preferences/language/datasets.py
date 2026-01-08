import random
from typing import Dict

import tensorflow as tf

from lm_human_preferences.datasets.books import books_generator
from lm_human_preferences.datasets.cnndm import cnndm_generator
from lm_human_preferences.datasets.tldr import tldr_generator

# 全局映射表(数据集名字->数据集对象)
_registry: Dict[str, "Dataset"] = {}

class Dataset:
    """
    一个用于构建和管理训练数据的数据集注册和生成类
    结合了注册表模式(Registry Pattern)、实例配置、TF数据流水线生成等最佳实践
    """
    def __init__(
        self,
        name,
        *,
        generator=None,
    ):
        """
        @name: 每个Dataset都要求有唯一name, 自动写入全局字典
        @generator: 生成器函数或对象
        """
        # 所有被初始化过的Dataset对象都自动被加进 _registry
        global _registry
        assert name not in _registry
        _registry[name] = self # 新建Dataset对象自动写入全局字典
        self.name = name
        self.generator = generator

    def tf_dataset(
        self,
        sequence_length,
        *,
        mode,
        encoder=None,
        seed=0,
        comm=None,
        shuffle=True,
        repeat_count=None,  # Defaults to infinite repeat
        start_token=None,   # trims so that it starts right after start token
        end_token=None,     # trims off last end_token
        padding_token=None,
    ):
        """
        将"生成器"包装成TF数据输入管道
        @sequence_length: 最终输出token长度
        @mode: 数据生成模式(如“train”、“test”)
        @encoder: tokenizer(含encode方法, 把文本转token id)
        @comm: 并行训练用的通信对象(如Horovod分布式训练, 可shard)
        @shuffle: 是否shuffle
        @repeat_count: epoch轮数, None代表无限
        @start_token/end_token/padding_token: 序列起始、终止和padding所需的特殊tokenid
        """
        print(f"debug. sequence_length: {sequence_length}, mode: {mode}, repeat_count: {repeat_count}")
        print(f"start_token: {start_token}, end_token: {end_token}, padding_token: {padding_token}")
        # debug. sequence_length: 64, mode: train, repeat_count: None
        # start_token: 13, end_token: 13, padding_token: None
        
        if padding_token is None:
            padding_token = encoder.padding_token
            print(f"padding_token: {padding_token}") # padding_token: 50259
            
        def _generator():
            inner_gen = self.generator(mode, seed=seed, shuffle=shuffle, comm=comm)
            for text in inner_gen:
                tokens = encoder.encode(text)
                # print(f"text: {text}, tokens: {tokens}, len(tokens): {len(tokens)}")
                """
                text: s o q t v. p q v b h. e o t a v. p m q n h w s h o x g o e. h a n t b. b g a k f g h e v y r i t l. v g x w i y q j e v q h x., 
                tokens: [82, 267, 10662, 256, 410, 13, 279, 10662, 410, 275, 289, 13, 304, 267, 256, 257, 410, 13, 279, 285, 10662, 299, 289, 266, 264, 289, 267, 2124, 308, 267, 304, 13, 289, 257, 299, 256, 275, 13, 275, 308, 257, 479, 277, 308, 289, 304, 410, 331, 374, 1312, 256, 300, 13, 410, 308, 2124, 266, 1312, 331, 10662, 474, 304, 410, 10662, 289, 2124, 13], 
                len(tokens): 67
                """
                
                # print(f"begin tokens: {tokens}")
                if start_token is not None:
                    try:
                        # 若设置start_token则只保留start_token之后部分
                        first_index = tokens.index(start_token) + 1
                        if first_index < len(tokens):
                            tokens = tokens[first_index:]
                    except:
                        continue

                # 只保留前n个token
                tokens = tokens[:sequence_length]

                # 如果有end_token, 从后往前截断, 只保留end_token之前的segment
                # print(f"end_token: {end_token}") # end_token: 13
                if end_token is not None:
                    try:
                        last_index = len(tokens) - tokens[::-1].index(end_token)
                        tokens = tokens[:last_index]
                    except:
                        continue

                # 不足sequence_length用padding_token对齐
                if len(tokens) < sequence_length:
                    tokens = tokens + [padding_token] * (sequence_length - len(tokens))

                # print(f"final tokens: {tokens}")
                # begin tokens: [75, 10662, 1976, 256, 304, 289, 479, 334, 474, 13, 256, 300, 275, 299, 1312, 264, 304, 264, 2124, 277, 285, 279, 277, 1312, 13, 266, 299, 2124, 289, 264, 267, 275, 479, 334, 13, 266, 334, 2124, 334, 308, 299, 2124, 257, 374, 1312, 374, 410, 264, 279, 308, 13, 334, 1312, 279, 1976, 479, 331, 10662, 1976, 285, 10662, 269, 1312, 13, 304, 269, 289, 267, 279, 13, 288, 267, 410, 2124, 266, 331, 289, 299, 289, 267, 269, 267, 308, 479, 2124, 13]
                # final tokens: [256, 300, 275, 299, 1312, 264, 304, 264, 2124, 277, 285, 279, 277, 1312, 13, 266, 299, 2124, 289, 264, 267, 275, 479, 334, 13, 266, 334, 2124, 334, 308, 299, 2124, 257, 374, 1312, 374, 410, 264, 279, 308, 13, 334, 1312, 279, 1976, 479, 331, 10662, 1976, 285, 10662, 269, 1312, 13, 304, 269, 289, 267, 279, 13, 50259, 50259, 50259, 50259]
                
                assert len(tokens) == sequence_length
                yield dict(tokens=tokens)

        # 包装为TF Dataset
        tf_dataset = tf.data.Dataset.from_generator(
            _generator,
            output_types=dict(tokens=tf.int32),
            output_shapes=dict(tokens=(sequence_length,)),
        )
        tf_dataset = tf_dataset.repeat(repeat_count)

        # 若处于分布式环境下, 自动分片shard, 只保留本节点用于训练的数据（如Horovod/TPU等环境）
        # 确保每台设备各自异步/平均处理一部分数据, 加快训练速度
        if comm is not None:
            num_shards = comm.Get_size()
            shard_idx = comm.Get_rank()
            if num_shards > 1:
                assert seed is not None
                tf_dataset = tf_dataset.shard(num_shards, shard_idx)
        return tf_dataset


def get_dataset(name) -> Dataset:
    """
    根据name动态获取已注册Dataset实例. 

    # 以下都会自动被注册到_registry
    dataset1 = Dataset("wiki", generator=wiki_generator)
    dataset2 = Dataset("lm1b", generator=lm1b_generator)
    # 可以直接通过name动态获取d, 即dataset1对象
    d = get_dataset("wiki")
    # 生成TF数据管道
    d.tf_dataset(...) 
    """
    global _registry
    return _registry[name]

# 注册各个dataset对象
CnnDm = Dataset("cnndm", generator=cnndm_generator,)
Tldr = Dataset("tldr", generator=tldr_generator,)
Books = Dataset("books", generator=books_generator,)

def test_generator(mode, seed=0, shuffle=False, comm=None):
    while True:
        yield ''.join([random.choice('abcdefghijklmnopqrstuvwxyz.') for _ in range(40)])
Test = Dataset("test", generator=test_generator)


"""
import tensorflow as tf
from lm_human_preferences.language.datasets import Books as ds
from lm_human_preferences.language.encodings import Main as encoding

e = encoding.get_encoder()
x = ds.tf_dataset(16, mode='test', encoder=e)
op = x.make_one_shot_iterator().get_next()
s = tf.Session()

while True:
    print(e.decode(s.run(op)['tokens']))
    input()
"""
