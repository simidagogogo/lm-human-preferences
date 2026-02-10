"""
Alec's transformer model.
"""

from functools import partial
from typing import Optional
from dataclasses import dataclass

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import function

from lm_human_preferences.utils import core as utils
from lm_human_preferences.utils import hyperparams


@dataclass
class HParams(hyperparams.HParams):
    # Encoding (set during loading process)
    n_vocab: int = 0

    # Model parameters
    n_ctx: int = 512
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12

    embd_pdrop: float = 0.1   # 嵌入Dropout
    attn_pdrop: float = 0.1   # 注意力Dropout
    resid_pdrop: float = 0.1  # 残差Dropout
    head_pdrop: float = 0.1   # 输出头Dropout


def parse_comma_separated_int_list(s):
    return [int(i) for i in s.split(",")] if s else []


def gelu(x):
    with tf.name_scope('gelu'):
        return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


def dropout(x, pdrop, *, do_dropout, stateless=True, seed=None, name):
    """
    Like tf.nn.dropout but stateless.
    @pdrop: 丢弃率, 1.0表示全部丢弃
    @do_dropout: True表示启用
    @stateless: True表示无状态, 即每次用同一个seed生成的dropout掩码完全相同(保证可复现性)
    """
    if stateless:
        assert seed is not None

    if pdrop == 0 or not do_dropout:
        return x

    def _dropout():
        with tf.name_scope(name):
            noise_shape = tf.shape(x)
            if stateless:
                r = tf.random.stateless_uniform(noise_shape, seed, dtype=x.dtype)
                # floor uniform [keep_prob, 1.0 + keep_prob), mask==1表示keep(keep=1-pdrop)
                mask = tf.floor(1 - pdrop + r)
                # 1/keep为补偿因子, 保证输出期望一致
                return x * (mask * (1 / (1 - pdrop)))
            else:
                return tf.nn.dropout(x, rate=pdrop, noise_shape=noise_shape)
    return _dropout()


def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """
    Normalize to mean = 0, std = 1, then do a diagonal affine transform.
    
    手动实现Layer Norm:
    1. 计算均值μ和方差σ^2
    2. 将输入x标准化为 x^hat(均值为0方差为1)
    3. 通过可学习参数g和b对x^hat进行仿射变换(缩放和平移)
    """
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        
        # gema
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        # beta
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        
        # E[x^2]
        s = tf.reduce_mean(tf.square(x), axis=axis, keepdims=True)
        # E[x]
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        
        # \sigma^2 = E[x^2] - (E[x])^2, 通常比tf.reduce_mean(tf.square(x - u))计算图结构稍微简单
        s = s - tf.square(u)
        
        # \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} 
        x = (x - u) * tf.rsqrt(s + epsilon)
        
        # y = \hat{x} \cdot g + b, 仿射变换让模型有能力重新调整数据分布(缩放和平移)
        x = x * g + b
        return x


def split_states(x, n):
    """
    Reshape the last dimension of x into [n, x.shape[-1]/n].
    From [batch, heads, features] to [batch, sequence, heads, head_dim]
    """
    *start, m = utils.shape_list(x)
    return tf.reshape(x, start + [n, m // n])


def merge_states(x):
    """
    Smash the last two dimensions of x into a single dimension.
    From [batch, sequence, heads, head_dim] to [batch, sequence, features]
    """
    *start, a, b = utils.shape_list(x)
    return tf.reshape(x, start + [a * b])


def conv1x1(x, scope, nf, *, w_init_stdev=0.02):
    """
    MLP
    """
    with tf.variable_scope(scope):
        *start, nx = utils.shape_list(x)
        # Don't cast params until just prior to use -- saves a lot of memory for large models
        with tf.control_dependencies([x]):
            w = tf.squeeze(
                    tf.get_variable(
                        'w', 
                        [1, nx, nf], 
                        initializer=tf.random_normal_initializer(stddev=w_init_stdev)
                    ), 
                axis=0
            )
            b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
            
        # TF1.x中tf.matmul仅支持二维矩阵相乘, 需要“三步”写法: "先reshape, 再乘, 再reshape回去”(TF2.x引入了tf.einsum和tf.tensordot)
        c = tf.matmul(tf.reshape(x, [-1, nx]), w) + b
        c = tf.reshape(c, start + [nf])
        return c


def attention_mask(nd, ns, *, dtype):
    """
    1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    
    # 确保解码阶段, 位置i的Query只能看到位置j≤i的Key, 而看不到“未来”的信息
    m = i >= j - ns + nd
    # to ignore first parts of context (useful for sampling with static shapes)
    # m = tf.math.logical_and(m, tf.math.logical_or(j  >= ignore, i < ignore - ns + nd))
    return tf.cast(m, dtype)


def softmax(x, axis=-1):
    """
    tf.nn.softmax()工业级别实现, 具有数值稳定性
    """
    # tf.exp(x-max(x))保证结果永远在[0, 1]之间, 解决inf/inf=NaN数值溢出问题
    # Softmax平移不变性: 给所有输入同时减去常数, 输出概率分布不变
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)


def attn(x, scope, hidden_dim, *, past, mask, do_dropout, scale=False, hparams, seed):
    """
    实现了支持增量推理(Incrumental Decoding)的高效多头注意力层
    不仅包含标准注意力计算(QKV投影->Attention->输出投影), 还通过past和present变量实现了KV Cache机制
    
    @past: 当前layer的kvcache. [batch, 2, heads, sequence, features], where 2 is [k, v]
    @mask: padding掩码. [batch, src_sequence]
    
    @return: 
        h: [batch, heads, sequence, features]
        present: 当前layer的kvcache. [batch, 2, heads, sequence, features]
    """
    # Should be [batch, sequence, features]
    assert x.shape.ndims == 3

    # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]
    if past is not None:
        assert past.shape.ndims == 5

    def split_heads(x):
        """
        From [batch, sequence, features] to [batch, heads, sequence, features]
        """
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        """
        Reverse of split_heads.
        From [batch, heads, sequence, features] to [batch, sequence, features]
        """
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        """
        注意力掩码, 包括因果掩码+padding掩码
        w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        """
        bs, _, nd, ns = utils.shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        
        if mask is not None:
            b *= tf.reshape(tf.cast(mask, w.dtype), [bs, 1, 1, ns])
        w = w * b - tf.cast(1e10, w.dtype) * (1 - b)
        return w

    def multihead_attn(q, k, v, *, seed):
        """
        MHA得分计算
        @q: [batch, heads, cur_sequence, features]
        @k: [batch, heads, all_sequence, features]
        @v: [batch, heads, all_sequence, features]
        @return: [batch, heads, cur_sequence, features]
        """
        # 在混合精度训练(Mixed Precision Training)中, 通常输入是float16或bfloat16
        # 为了保证Attention计算(点积、Softmax)的数值稳定性, 防止溢出, 通常会把这几个关键变量临时转成float32进行运算
        orig_dtype = v.dtype
        q, k, v = map(partial(tf.cast, dtype=tf.float32), (q, k, v))
        
        # [batch, heads, cur_sequence, all_sequence]
        w = tf.matmul(q, k, transpose_b=True)

        # 防止梯度消失
        if scale:
            hidden_dim = v.shape[-1].value
            w = w * tf.rsqrt(tf.cast(hidden_dim, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        # droput位置: 注意力
        w = dropout(w, hparams.attn_pdrop, do_dropout=do_dropout, name='attn_drop', stateless=True, seed=seed)
        
        # [batch, heads, cur_sequence, features]
        a = tf.matmul(w, v)
        a = tf.cast(a, dtype=orig_dtype, name='a_cast')
        return a

    with tf.variable_scope(scope):
        attn_seed, resid_seed = split_seed(seed, 2)
        assert hidden_dim % hparams.n_head == 0
        w_init_stdev = 1 / np.sqrt(hidden_dim)
        
        # 1个qkv大矩阵 -> 3个多头qkv矩阵
        c = conv1x1(x, 'c_attn', hidden_dim * 3, w_init_stdev=w_init_stdev)
        # q, k, v: [batch, heads, sequence, features]
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        
        # 当前layer的kvcache, 对应与当前输入的X
        # [batch, 2, heads, sequence, features]
        present = tf.stack([k, v], axis=1)
        
        # 拼接kvcache
        if past is not None:
            # [batch, heads, sequence, features],
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        
        # [batch, heads, sequence, features]
        a = multihead_attn(q, k, v, seed=attn_seed)
        # [batch, sequence, features]
        a = merge_heads(a)
        
        w_init_stdev = 1 / np.sqrt(hidden_dim * hparams.n_layer)
        a = conv1x1(a, 'c_proj', hidden_dim, w_init_stdev=w_init_stdev)
        # droput位置: 输出头
        a = dropout(a, hparams.resid_pdrop, do_dropout=do_dropout, stateless=True, seed=resid_seed, name='attn_resid_drop')
        return a, present


def mlp(x, scope, n_hidden, *, do_dropout, hparams, seed):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        w_init_stdev = 1 / np.sqrt(nx)
        h = gelu(conv1x1(x, 'c_fc', n_hidden, w_init_stdev=w_init_stdev))
        
        w_init_stdev = 1 / np.sqrt(n_hidden * hparams.n_layer)
        h2 = conv1x1(h, 'c_proj', nx, w_init_stdev=w_init_stdev)
        # droput位置: 输出头
        h2 = dropout(h2, hparams.resid_pdrop, do_dropout=do_dropout, stateless=True, seed=seed, name='mlp_drop')
        return h2


def block(x, scope, *, past, mask, do_dropout, scale=False, hparams, seed):
    """
    Transformer block
    @past: 当前层的past
    @return: 
        h: [batch, sequence, features]
        present: [batch, 2, heads, sequence, features]
    """
    with tf.variable_scope(scope):
        attn_seed, mlp_seed = split_seed(seed, 2)
        hidden_dim = x.shape[-1].value
        
        # 1. attention sublayer
        a, present = attn(
            norm(x, 'ln_1'), # norm位置: pre_attn, ln_1
            'attn', 
            hidden_dim, 
            past=past, 
            mask=mask, 
            do_dropout=do_dropout, 
            scale=scale, 
            hparams=hparams, 
            seed=attn_seed
        )
        # residual layer
        x = x + a

        # 2. ffn sublayer
        m = mlp(
            norm(x, 'ln_2'), # norm位置: pre_ffn, ln_2
            'mlp', 
            hidden_dim * 4, 
            do_dropout=do_dropout, 
            hparams=hparams, 
            seed=mlp_seed
        )
        # residual layer
        h = x + m
        return h, present


# TODO
@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()]
)
def convert_gradient_to_tensor(x):
    """
    Force gradient to be a dense tensor.
    It's often faster to do dense embedding gradient on GPU than sparse on CPU.
    """
    return x


def embed(X, we):
    """
    Embedding lookup.
    X has shape [batch, sequence, info].  Currently info = 2 corresponding to [token_id, position].
    """
    we = convert_gradient_to_tensor(we)
    e = tf.gather(we, X)
    return e


def tensordot(x, y, num_axes):
    """
    tf.matmul()高维推广
    tensor contraction of the final axes of x with the first axes of y need to write it ourselves 
    because tensorflow's tf.tensordot() is slow
    """
    split_x_axes_at = x.shape.ndims - num_axes
    
    # first axes of x
    x_shape = tf.shape(x)[:split_x_axes_at]
    # final axes of y
    y_shape = tf.shape(y)[num_axes:]
    
    rx = tf.reshape(x, [tf.reduce_prod(x_shape), tf.reduce_prod(tf.shape(x)[split_x_axes_at:])])
    ry = tf.reshape(y, [-1, tf.reduce_prod(y_shape)])
    rresult = tf.matmul(rx, ry)
    result = tf.reshape(rresult, tf.concat([x_shape, y_shape], axis=0))
    
    # 设置静态shape, 静态类型推断有利于调试、优化和静态检查
    result.set_shape(x.shape[:split_x_axes_at].concatenate(y.shape[num_axes:]))
    return result


def fc_layer(x, outshape, *, in_axes=1, scale=None):
    """
    more convenient fc layer that avoids stupid shape stuff consumes in_axes of x produces y of shape outshape
    
    @x: [batch, sequence, model_d]
    @outshape: 
    @in_axes: 指定输入张量x的最后几维参与运算. 全连接层只处理最后一维in_axes=1, 允许把最后N维展平后一起作为输入特征
    @return 
        tensordot(x, w, in_axes) + b: 前向输出
        reg_loss: L2正则化损失, 用于加到总Loss中. Loss_{total} = Loss_{data} + \lambda \cdot \sum w^2
        
    """
    inshape = tuple([int(d) for d in x.shape[-in_axes:]]) if in_axes > 0 else ()
    outshape = tuple(outshape)
    
    # xavier/glorot初始化变体: 1/sqrt(Fan_in+1), Fan-in输入特征的总数量.
    # 保持每一层输出的方差一致, 保证初始梯度流动平稳, 不易爆炸或消失
    if scale is None:
        scale = 1 / np.sqrt(np.prod(inshape) + 1)
    
    # (model_d, )
    w = tf.get_variable('w', inshape + outshape, initializer=tf.random_normal_initializer(stddev=scale))
    # ()
    b = tf.get_variable('b', outshape, initializer=tf.constant_initializer(0))
    
    """
    L2正则化损失: 权重所有元素的平方和 乘以 正则化系数1/Fan_out
    当输出维度很大时, 权重矩阵W里元素非常多. 如果scale不变, 累加起来的正则化Loss会变得非常大, 导致模型根本没法学(Loss 被正则项主导)
    通过除以元素个数(或输出个数), 是为了让正则化项的初始值保持在一个相对稳定、与网络规模无关的量级上(大约为1左右, 如代码注释所言)
    """
    # Call the regularizer manually so that it works correctly with GradientTape
    # so that initial value of regularizer is 1
    regularizer = tf.contrib.layers.l2_regularizer(scale=1/np.prod(outshape))
    reg_loss = regularizer(w)
    return tensordot(x, w, in_axes) + b, reg_loss


def past_shape(*, hparams, batch_size=None, sequence=None):
    """
    kv Cache. (batch_size, n_layer, 2, n_head, sequence, head_dim)
    """
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, utils.exact_div(hparams.n_embd, hparams.n_head)]


def positions_for(*, batch, sequence, past_length, mask):
    """
    计算位置编码position值
    :param batch: 批量大小
    :param sequence: 当前序列长度
    :param past_length: 历史序列长度
    :param mask: padding掩码
    """
    if mask is None:
        return utils.expand_tile(past_length + tf.range(sequence), batch, axis=0)
    else:
        return tf.cumsum(tf.cast(mask, tf.int32), exclusive=True, axis=-1)[:, past_length:]


def split_seed(seed, n=2):
    """
    随机种子生成器. 无状态保证同样的seed产生同样的结果
    :param seed: 种子
    :param n: 分割数量
    :return: 分割后的种子
    """
    if n == 0:
        return []

    return tf.split(
        tf.random.stateless_uniform(dtype=tf.int64, shape=[2*n], minval=-2**63, maxval=2**63-1, seed=seed),
        n, 
        name='split_seeds'
    )


class Model:
    def __init__(self, hparams: HParams, scalar_heads=[], scope=None):
        self.hparams = hparams
        
        # 支持普通输出 和 value head(强化学习中价值预测)
        self.scalar_heads = scalar_heads
        with tf.variable_scope(scope, 'model') as scope:
            self.scope = scope
        self.built = False


    def __call__(
        self, 
        *, 
        X, 
        Y=None, 
        past=None, 
        past_tokens=None, 
        mask=None,
        padding_token: Optional[int]=None, 
        do_dropout=False
    ):
        """
        模型前向传播
        :param X: 输入tokens
        :param Y: unused
        :param past: kv缓存
        :param past_tokens: 所有历史tokens
        """
        X = tf.convert_to_tensor(X, dtype=tf.int32)
        if mask is not None:
            mask = tf.convert_to_tensor(mask, dtype=tf.bool)
            assert mask.dtype == tf.bool
        
        # 填充token处理
        if padding_token is not None:
            assert mask is None, 'At most one of mask and padding_token should be set'
            mask = tf.not_equal(X, padding_token)
            X = tf.where(mask, X, tf.zeros_like(X))
            if past is not None:
                assert past_tokens is not None, 'padding_token requires past_tokens'
                mask = tf.concat([tf.not_equal(past_tokens, padding_token), mask], axis=1)
        
        with tf.variable_scope(
            self.scope, 
            reuse=self.built, 
            auxiliary_name_scope=not self.built
        ):
            self.built = True
            results = {}
            batch, sequence = utils.shape_list(X)
            seed = tf.random.uniform(dtype=tf.int64, shape=[2], minval=-2**63, maxval=2**63-1)
            wpe_seed, wte_seed, blocks_seed, heads_seed = split_seed(seed, 4)

            # 位置编码权重词表
            wpe = tf.get_variable(
                'wpe', 
                [self.hparams.n_ctx, self.hparams.n_embd],
                initializer=tf.random_normal_initializer(stddev=0.01)
            )
            
            # tokenemb权重词表
            wte = tf.get_variable(
                'wte', 
                [self.hparams.n_vocab, self.hparams.n_embd],
                initializer=tf.random_normal_initializer(stddev=0.02)
            )
            
            # dropout位置: token_emb
            wpe = dropout(
                wpe, 
                self.hparams.embd_pdrop,
                do_dropout=do_dropout, 
                stateless=True,
                seed=wpe_seed, 
                name='wpe_drop'
            )
            
            # dropout位置: pos_emb
            wte = dropout(
                wte, 
                self.hparams.embd_pdrop,
                do_dropout=do_dropout, 
                stateless=True, 
                seed=wte_seed, 
                name='wte_drop'
            )

            past_length = 0 if past is None else tf.shape(past)[-2]
            positions = positions_for(
                batch=batch, 
                sequence=sequence, 
                past_length=past_length, 
                mask=mask
            )
            
            # [batch, sequence, token_emb] + [sequence, pos_emb]
            h = embed(X, wte) + embed(positions, wpe)
            
            # Transformer
            presents = []
            # 拆分每层KV cache
            pasts = tf.unstack(past, axis=1) if past is not None else [None] * self.hparams.n_layer
            assert len(pasts) == self.hparams.n_layer
            block_seeds = split_seed(blocks_seed, self.hparams.n_layer)
            
            for layer, (past, block_seed) in enumerate(zip(pasts, block_seeds)):
                h, present = block(
                    h, 
                    'h%d' % layer, 
                    past=past, 
                    mask=mask, 
                    do_dropout=do_dropout, 
                    scale=True,
                    hparams=self.hparams, 
                    seed=block_seed
                )
                # present: [batch, 2, heads, sequence, features]
                presents.append(present)
            # present: [batch, layers, 2, heads, sequence, features]
            results['present'] = tf.stack(presents, axis=1)
            
            # norm位置: 计算logits之前, ln_f
            h = norm(h, 'ln_f')
            if mask is not None:
                # For non-present tokens, use the output from the last present token instead.
                present_indices = utils.where(
                    mask[:, past_length:], 
                    tf.tile(tf.range(sequence)[None,:], [batch, 1]), 
                    -1
                )
                use_indices = utils.cumulative_max(present_indices)
                # assert since GPUs don't
                with tf.control_dependencies([tf.assert_none_equal(use_indices, -1)]):
                    h = utils.index_each(h, use_indices)
            
            # [batch, sequence, token_emb]
            results['h'] = h

            # Language model loss. Do tokens <n predict token n?
            h_flat = tf.reshape(h, [batch * sequence, self.hparams.n_embd])
            flat_lm_logits = tf.matmul(h_flat, wte, transpose_b=True)

            labels = tf.concat([X[:, 1:], X[:, :1]], axis=1)
            flat_labels = tf.reshape(labels, [batch * sequence])

            flat_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=flat_labels,
                logits=flat_lm_logits
            )

            # [batch, sequence]
            lm_losses = tf.reshape(flat_losses, [batch, sequence])
            # [batch, sequence, vocab_n]
            lm_logits = tf.reshape(flat_lm_logits, [batch, sequence, -1])

            # [batch, sequence-1]
            relevant_losses = lm_losses[:, :-1]
            results['lm_all_losses'] = relevant_losses
            # [batch, sequence, vocab_n]
            results['lm_logits'] = lm_logits
            # (batch, )
            results['lm_losses'] = tf.reduce_mean(relevant_losses, axis=-1)

            head_seeds = split_seed(heads_seed, len(self.scalar_heads))
            for head_name, head_seed in zip(self.scalar_heads, head_seeds):
                with tf.variable_scope(f"heads/{head_name}"):
                    # dropout位置: 输出头
                    dropped_h = dropout(h, self.hparams.head_pdrop, do_dropout=do_dropout, seed=head_seed, name='drop')
                    # TODO: refactor this, perhaps move to Policy
                    # 标量值 和 l2正则化值
                    # PPO/RLHF训练技巧: 构建价值函数Value Head, 权重初始化为全0; 其他功能头(比如Reward Head), 使用随机初始化
                    # res: (batch, sequence)
                    # reg_loss: ()
                    res, reg_loss = fc_layer(dropped_h, (), scale=0 if head_name == 'value' else None)
                    results[head_name] = tf.cast(res, dtype=tf.float32, name='res_cast')
                    results[f"{head_name}_regularizer"] = tf.cast(reg_loss, dtype=tf.float32, name='reg_loss_cast')
            
            # All done!
            print(f"[__call__]. results: {results}")
            """
            [__call__]. results: {
                'present': <tf.Tensor 'step/ref_policy/model/stack:0' shape=(?, 12, 2, 12, ?, 64) dtype=float32>, 
                'h': <tf.Tensor 'step/ref_policy/model/index_each:0' shape=(?, ?, 768) dtype=float32>, 
                'lm_all_losses': <tf.Tensor 'step/ref_policy/model/strided_slice_7:0' shape=(?, ?) dtype=float32>, 
                'lm_logits': <tf.Tensor 'step/ref_policy/model/Reshape_3:0' shape=(?, ?, ?) dtype=float32>, 
                'lm_losses': <tf.Tensor 'step/ref_policy/model/Mean:0' shape=(?,) dtype=float32>, 
                'value': <tf.Tensor 'step/ref_policy/model/heads/value/add:0' shape=(?, ?) dtype=float32>, 
                'value_regularizer': <tf.Tensor 'step/ref_policy/model/heads/value/l2_regularizer:0' shape=() dtype=float32>
            }
            [__call__]. results: {
                'present': <tf.Tensor 'reward_model/model_1/stack:0' shape=(?, 12, 2, 12, ?, 64) dtype=float32>, 
                'h': <tf.Tensor 'reward_model/model_1/index_each:0' shape=(?, ?, 768) dtype=float32>, 
                'lm_all_losses': <tf.Tensor 'reward_model/model_1/strided_slice_7:0' shape=(?, ?) dtype=float32>, 
                'lm_logits': <tf.Tensor 'reward_model/model_1/Reshape_3:0' shape=(?, ?, ?) dtype=float32>, 
                'lm_losses': <tf.Tensor 'reward_model/model_1/Mean:0' shape=(?,) dtype=float32>, 
                'reward': <tf.Tensor 'reward_model/model_1/heads/reward/add:0' shape=(?, ?) dtype=float32>, 
                'reward_regularizer': <tf.Tensor 'reward_model/model_1/heads/reward/l2_regularizer:0' shape=() dtype=float32>
            }
            [__call__]. results: {
                'present': <tf.Tensor 'train_reward/minibatch/stack:0' shape=(?, 12, 2, 12, 88, 64) dtype=float32>, 
                'h': <tf.Tensor 'train_reward/minibatch/index_each:0' shape=(?, 88, 768) dtype=float32>, 
                'lm_all_losses': <tf.Tensor 'train_reward/minibatch/strided_slice_6:0' shape=(?, 87) dtype=float32>, 
                'lm_logits': <tf.Tensor 'train_reward/minibatch/Reshape_3:0' shape=(?, 88, ?) dtype=float32>, 
                'lm_losses': <tf.Tensor 'train_reward/minibatch/Mean:0' shape=(?,) dtype=float32>, 
                'reward': <tf.Tensor 'train_reward/minibatch/heads/reward/add:0' shape=(?, 88) dtype=float32>, 
                'reward_regularizer': <tf.Tensor 'train_reward/minibatch/heads/reward/l2_regularizer:0' shape=() dtype=float32>
            }
            [__call__]. results: {
                'present': <tf.Tensor 'train_reward/minibatch/stack_1:0' shape=(?, 12, 2, 12, 88, 64) dtype=float32>, 
                'h': <tf.Tensor 'train_reward/minibatch/index_each_1:0' shape=(?, 88, 768) dtype=float32>, 
                'lm_all_losses': <tf.Tensor 'train_reward/minibatch/strided_slice_14:0' shape=(?, 87) dtype=float32>, 
                'lm_logits': <tf.Tensor 'train_reward/minibatch/Reshape_7:0' shape=(?, 88, ?) dtype=float32>, 
                'lm_losses': <tf.Tensor 'train_reward/minibatch/Mean_1:0' shape=(?,) dtype=float32>, 
                'reward': <tf.Tensor 'train_reward/minibatch/heads/reward_1/add:0' shape=(?, 88) dtype=float32>, 
                'reward_regularizer': <tf.Tensor 'train_reward/minibatch/heads/reward_1/l2_regularizer:0' shape=() dtype=float32>
            }
            [__call__]. results: {
                'present': <tf.Tensor 'train_reward/minibatch/stack_2:0' shape=(?, 12, 2, 12, 88, 64) dtype=float32>, 
                'h': <tf.Tensor 'train_reward/minibatch/index_each_2:0' shape=(?, 88, 768) dtype=float32>, 
                'lm_all_losses': <tf.Tensor 'train_reward/minibatch/strided_slice_22:0' shape=(?, 87) dtype=float32>, 
                'lm_logits': <tf.Tensor 'train_reward/minibatch/Reshape_11:0' shape=(?, 88, ?) dtype=float32>, 
                'lm_losses': <tf.Tensor 'train_reward/minibatch/Mean_2:0' shape=(?,) dtype=float32>, 
                'reward': <tf.Tensor 'train_reward/minibatch/heads/reward_2/add:0' shape=(?, 88) dtype=float32>, 
                'reward_regularizer': <tf.Tensor 'train_reward/minibatch/heads/reward_2/l2_regularizer:0' shape=() dtype=float32>}
            [__call__]. results: {
                'present': <tf.Tensor 'train_reward/minibatch/stack_3:0' shape=(?, 12, 2, 12, 88, 64) dtype=float32>, 
                'h': <tf.Tensor 'train_reward/minibatch/index_each_3:0' shape=(?, 88, 768) dtype=float32>, 
                'lm_all_losses': <tf.Tensor 'train_reward/minibatch/strided_slice_30:0' shape=(?, 87) dtype=float32>, 
                'lm_logits': <tf.Tensor 'train_reward/minibatch/Reshape_15:0' shape=(?, 88, ?) dtype=float32>, 
                'lm_losses': <tf.Tensor 'train_reward/minibatch/Mean_3:0' shape=(?,) dtype=float32>, 
                'reward': <tf.Tensor 'train_reward/minibatch/heads/reward_3/add:0' shape=(?, 88) dtype=float32>, 
                'reward_regularizer': <tf.Tensor 'train_reward/minibatch/heads/reward_3/l2_regularizer:0' shape=() dtype=float32>}
            [__call__]. results: {
                'present': <tf.Tensor 'sample_seq/step/stack:0' shape=(512, 12, 2, 12, 64, 64) dtype=float32>, 
                'h': <tf.Tensor 'sample_seq/step/index_each:0' shape=(512, 64, 768) dtype=float32>, 
                'lm_all_losses': <tf.Tensor 'sample_seq/step/strided_slice_5:0' shape=(512, 63) dtype=float32>, 
                'lm_logits': <tf.Tensor 'sample_seq/step/Reshape_3:0' shape=(512, 64, 50257) dtype=float32>, 
                'lm_losses': <tf.Tensor 'sample_seq/step/Mean:0' shape=(512,) dtype=float32>, 
                'value': <tf.Tensor 'sample_seq/step/heads/value/add:0' shape=(512, 64) dtype=float32>, 
                'value_regularizer': <tf.Tensor 'sample_seq/step/heads/value/l2_regularizer:0' shape=() dtype=float32>}
            [__call__]. results: {
                'present': <tf.Tensor 'sample_seq/while/step/stack:0' shape=(512, 12, 2, 12, ?, 64) dtype=float32>, 
                'h': <tf.Tensor 'sample_seq/while/step/index_each:0' shape=(512, 1, 768) dtype=float32>, 
                'lm_all_losses': <tf.Tensor 'sample_seq/while/step/strided_slice_6:0' shape=(512, 0) dtype=float32>, 
                'lm_logits': <tf.Tensor 'sample_seq/while/step/Reshape_3:0' shape=(512, 1, 50257) dtype=float32>, 
                'lm_losses': <tf.Tensor 'sample_seq/while/step/Mean:0' shape=(512,) dtype=float32>, 
                'value': <tf.Tensor 'sample_seq/while/step/heads/value/add:0' shape=(512, 1) dtype=float32>, 
                'value_regularizer': <tf.Tensor 'sample_seq/while/step/heads/value/l2_regularizer:0' shape=() dtype=float32>}
            """
            return results


    def get_params(self):
        """
        确保当前模型对象的计算图(神经网络结构和变量)已经完全构建完成
        """
        assert self.built
        params = utils.find_trainable_variables(self.scope.name)
        assert len(params) > 0
        return params
