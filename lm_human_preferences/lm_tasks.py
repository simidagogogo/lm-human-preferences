from dataclasses import dataclass, field
from typing import Optional

import tensorflow as tf

from lm_human_preferences.language import datasets      # 导入datasets.py文件
from lm_human_preferences.utils import core as utils    # 导入core.py文件
from lm_human_preferences.utils import hyperparams      # 导入hyperparams.py文件


@dataclass
class PolicyHParams(hyperparams.HParams):
    initial_model: str = None # 模型名称: 124M
    temperature: float = 1.0  # 调节概率分布平滑度. >1概率更均匀, <1概率更集中 


@dataclass
class TaskHParams(hyperparams.HParams):
    # 1. Query params
    # Query长度. 通常指定每次输入模型的查询序列长度(如prompt有多少个token)
    query_length: int = None
    
    # Query数据集名称. 比如用哪个数据集生成查询, 如"books"
    query_dataset: str = None
    
    # 每次查询前面添加的内容. 通常用于prompt engineering定制化
    # 通常前缀如"<User>: ", "Question: "等固定文本
    query_prefix: str = ''
    
    # 每次查询后面附加的内容. 同样用于prompt engineering定制化
    # 通常后缀如"<EOS>", 某种结束标记
    query_suffix: str = ''
    
    # Query起始符号/字符串. 用于标识输入开始, Optional表示可以为None
    start_text: Optional[str] = '.'
    
    # Query结束符号/字符串. 用于标识输入query终止
    end_text: Optional[str] = None

    # 2. Response params
    # 期望模型输出响应的长度
    response_length: int = None

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    # 若指定了这个token, 则采样时在“index after”之后, 遇到第一个该token时截断响应(适合终止符控制)
    truncate_token: Optional[int] = None
    
    # 从哪个index之后才能截断响应(配合truncate_token使用)
    truncate_after: int = 0
    
    # 惩罚/奖励的数值(比如在强化学习中, 遇到某种情况返回这个分数)
    # PPO算法中KL_penalty常用Reference Policy做为KL基准, 鼓励新策略不要偏离原始模型太远
    penalty_reward_value: int = -1
    
    # 用于存储具体“策略”的超参数
    policy: PolicyHParams = field(default_factory=PolicyHParams)


def postprocess_fn_from_hparams(hparams: TaskHParams, padding_token: int):
    """
    #returns a postprocessing function
    #it is applied to responses before they are scored
    #central example: replace all tokens after truncate_token with padding_token
    """
    def get_mask(responses, truncate_token, truncate_after):
        # We want to truncate at the first occurrence of truncate_token that appears at or after
        # position truncate_after in the responses
        mask = tf.cast(tf.equal(responses, truncate_token), tf.int32)
        mask = tf.concat([tf.zeros_like(mask)[:,:truncate_after], mask[:,truncate_after:]], axis=1)
        return tf.cast(tf.cumsum(mask, axis=1) - mask, tf.bool)
    if hparams.truncate_token is not None:
        def truncate(responses):
            mask = get_mask(responses, hparams.truncate_token, hparams.truncate_after)
            return tf.where(mask, padding_token * tf.ones_like(responses), responses)
        return truncate
    else:
        return lambda responses: responses


def filter_fn_from_hparams(hparams: TaskHParams):
    """
    #returns a filter function
    #responses not passing that function will receive a low (fixed) score
    #only query humans on responses that pass that function
    #central example: ensure that the sample contains truncate_token
    """
    def filter(responses):
        if hparams.truncate_token is not None:
            matches_token = tf.equal(responses[:, hparams.truncate_after:], hparams.truncate_token)
            return tf.reduce_any(matches_token, axis=-1)
        else:
            return tf.ones(tf.shape(responses)[0], dtype=tf.bool)
    return filter


def query_formatter(hparams: TaskHParams, encoder):
    """Turns a query into a context to feed to the language model
    将输入queries连同前缀和后缀拼接成完整的上下文(prompt)输入喂给LLM. 可针对不同任务自定义开头、结尾的提示token
    NOTE: Both of these are lists of tokens

    用于dialogue、问答、条件文本生成等任务, 一般需要将一些prompt、query以及特殊符号拼接后统一投喂给LLM. 例子:
        hparams.query_prefix: '问题:'
        hparams.query_suffix: '<EOS>'
        queries: batch的每条形如 ['狗能吃巧克力吗', ...]
        
        encoder.encode(): 
            ['问题:'] → [102,203]
            ['<EOS>'] → [1]
            queries  →token序列
        
        最终模型输入:
            [102,203, x1,x2,...,xn, 1]
            [102,203, y1,y2,...,ym, 1]

    :param hparams: 
    :param encoder: 可逆分词器与编码器(ReversibleEncoder类的实例)
    """

    def query_formatter(queries):
        """将query格式化
        :param queries: 查询, shape: [batch_size, query_len]
        :return: 横向拼接[前缀, 查询, 后缀], shape: [batch_size, prefix_len + query_len + suffix_len]
        """
        batch_size = tf.shape(queries)[0]
        prefix_tokens = tf.constant(encoder.encode(hparams.query_prefix), dtype=tf.int32)
        tiled_prefix = utils.expand_tile(prefix_tokens, batch_size, axis=0)
        print(f"tiled_prefix.shape: {tiled_prefix.shape.as_list()}")

        suffix_tokens = tf.constant(encoder.encode(hparams.query_suffix), dtype=tf.int32)
        tiled_suffix = utils.expand_tile(suffix_tokens, batch_size, axis=0)
        print(f"tiled_suffix.shape: {tiled_suffix.shape.as_list()}")
        return tf.concat([tiled_prefix, queries, tiled_suffix], 1)
    return query_formatter


def make_query_sampler(*, hparams: TaskHParams, encoder, batch_size: int, mode='train', comm=None):
    """
    :param encoder: 可逆分词器与编码器(ReversibleEncoder类的实例)
    :return: 
    """
    if hparams.start_text:
        # 解包: 既完成赋值(期望start_token为整数, 而不是列表), 也隐式地断言右边必须且只能有一个元素
        start_token, = encoder.encode(hparams.start_text)
    else:
        start_token = None

    if hparams.end_text:
        end_token, = encoder.encode(hparams.end_text)
    else:
        end_token = None

    # 获取dataset
    dataset = datasets.get_dataset(hparams.query_dataset)
    
    # 随时待命的数据源
    data = dataset.tf_dataset(
        sequence_length=hparams.query_length, 
        mode=mode, 
        comm=comm, 
        encoder=encoder,
        start_token=start_token, 
        end_token=end_token,
    )

    # 将字典数据{tokens: [12, 42, 57]}变成类型tf.int32的列表[12, 42, 57]
    data = data.map(lambda d: tf.cast(d['tokens'], tf.int32))
    data = data.batch(batch_size, drop_remainder=True)
    context_iterator = data.make_one_shot_iterator()

    def sampler(scope=None):
        """
        :param scope: "ref_queries"
        :return: 
        """
        with tf.name_scope(scope, 'sample_corpus'):
            context_tokens = context_iterator.get_next()
            return dict(tokens=context_tokens)  # key为: "tokens"
    return sampler
