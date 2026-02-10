#!/usr/bin/env python3
"""
Test sample_sequence().
~/PycharmProjects/lm-human-preferences/lm_human_preferences/language# pipenv run ./test_sample.py 

(base) root@iZ0jlfyn5du7ptefx2tr5vZ:~/PycharmProjects/lm-human-preferences/lm_human_preferences/language# pipenv run ./test_sample.py 
first_output_logits.shape: (2, 10)
first_outputs.shape: (2,)
first_logprobs.shape: (2,)
WARNING:tensorflow:From /root/.local/share/virtualenvs/lm-human-preferences-XpxZn-hG/lib/python3.7/site-packages/tensorflow/python/ops/control_flow_ops.py:423: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
logits.shape: (2, 1, 10), next_sample.shape: (2, 1), next_logprob: Tensor("sample_seq/while/Neg:0", shape=(2, 1), dtype=float32)
2026-02-10 21:01:17.480740: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2026-02-10 21:01:17.501726: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2699995000 Hz
2026-02-10 21:01:17.501939: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x705a0c0 executing computations on platform Host. Devices:
2026-02-10 21:01:17.501958: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
(base) root@iZ0jlfyn5du7ptefx2tr5vZ:~/PycharmProjects/lm-human-preferences/lm_human_preferences/language# 
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

from lm_human_preferences.language import sample

n_vocab = 10
batch_size = 2
hparams = HParams(
    n_layer=0,
    n_head=1,
    n_embd=0,
    n_attn=0,
)

# Returns a policy that deterministically chooses previous token + 1.
def step(hparams, tokens, past=None, past_tokens=None):
    """
    step函数并不是一个真正的神经网络模型. 作用是确定性地生成下一个token. 
    具体规则: 如果当前输入的token是N, 那么下一个预测的token必须是 N+1. 这通常用于单元测试, 验证训练循环、数据管道或奖励机制是否正常工作, 而无需加载庞大且计算昂贵的真实模型
    
    @hparams: 超参数
    @tokens: 当前步的输入
    @past: 缓存了之前所有时间步计算出的Key和Value矩阵. 模型只需要计算当前最新输入token的Key和Value, 然后拼接到缓存中即可(O(1)常数级)
    @past_tokens: 之前所有的历史输入(有时也包括提示词Prompt)
    """
    logits = tf.one_hot(tokens + 1, n_vocab, on_value=0., off_value=-np.inf, dtype=tf.float32)
    ret = {
        'logits': logits,
        'presents': tf.zeros(shape=[2, 0, 2, 1, 0, 0]),     # [batch_size, num_layers, 2, num_heads, sequence_length, head_emb_size]
    }
    return ret

def test_sample_sequence():
    output = sample.sample_sequence(
        step=step, 
        model_hparams=hparams, 
        length=4, 
        batch_size=batch_size,
        context=tf.constant([[5, 0], [4, 3]])
    )
    expected = np.array([[5, 0, 1, 2, 3, 4], [4, 3, 4, 5, 6, 7]])
    with tf.Session() as sess:
        np.testing.assert_array_equal(sess.run(output)['tokens'], expected)


if __name__ == '__main__':
    test_sample_sequence()
