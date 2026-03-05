#!/usr/bin/env python3

import tensorflow as tf
import random

from lm_human_preferences.language.datasets import Dataset
from lm_human_preferences.language.datasets import Books as ds
from lm_human_preferences.language.encodings import Main as encoding

def test_generator(mode, seed=0, shuffle=False, comm=None):
    """无限数据生成器. 不断产出由随机小写字母和点号组成的长度为40的乱码字符串
    :param mode:
    :param seed:
    :param shuffle: 
    :parm comm:
    :return: 
    """
    while True:
        yield ''.join([random.choice('abcdefghijklmnopqrstuvwxyz.') for _ in range(40)])
Test = Dataset("test", generator=test_generator)

"""
数据浏览器Data Viewer. 工作流程：
1. 准备好书本数据集和编码器。
2. 建立一个批大小为 16 的数据管道
3. 每次按回车就从管道里拉取一批数据(token_id)
4. 把token_id解码成文字并打印到屏幕上, 让你检查数据格式是否正确, 或者查看数据集里的内容是什么
"""

# 返回ReversibleEncoder类的实例, 一个可逆分词器与编码器
e = encoding.get_encoder()
# x是一个TF计算图节点(Tensor), 代表这个数据流
x = ds.tf_dataset(16, mode='test', encoder=e)
# make_one_shot_iterator创建一个迭代器(从头到尾遍历一次数据集), get_next定义一个计算图操作Op. 
# 不会立即读取数据, 只是在计算图中定义了一个“取数”动作. 每次在Session中运行op就会从管道里吐出一批新的数据(这里batch_size=1)
op = x.make_one_shot_iterator().get_next()
# 启动TF会话, 分配资源准备计算
s = tf.Session()

while True:
    # s.run()真正执行“取数”动作
    print(e.decode(s.run(op)['tokens']))
    # 程序暂停等待用户按回车键. 按一次回车, 查看下一批数据
    input()

"""
(.venv)  ✘ ⚙ xiangqian@U-YXH5W5CQ-0216  ~/PycharmProjects/lm-human-preferences/lm_human_preferences/language   master ±  pipenv run python3.7 ./test_datasets.py 
"""