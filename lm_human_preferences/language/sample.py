import tensorflow as tf

from lm_human_preferences.language import model
from lm_human_preferences.utils import core as utils

def sample_sequence(
    *, 
    step, 
    model_hparams, 
    length, 
    batch_size=None, 
    context=None,
    temperature=1, 
    top_k=0, 
    top_p=1.0, 
    extra_outputs={}, 
    cond=None
):
    """
    Sampling from an autoregressive sequence model.
    自回归文本生成(解码阶段). 接收一段提示语Context, 利用训练好的模型(Step函数), 一个字一个字(Token by Token)地往后写, 直到写满指定长度.
    
    @step: A function(训练好模型) which takes model hparams, a tokens Tensor, past, and returns a dictionary with 'logits' and 'presents', and any extra vars.
    @context: Includes start tokens. (batch_size, sequence)
    @temperature: 随机度控制. >1更随机(胡言乱语), <1 更保守(重复单调). 代码里通过调整logits实现
    @top_k: 仅从概率最高K个词中采样. 防止生成极其离谱的词
    @top_p: 核采样. 仅从累积概率达到P的词集合中采样. 比Top-K更动态、自然
    @extra_outputs: Map from extra output key to dtype
    Returns: A dict with keys 'presents', 'logits', and any keys in extra_outputs
    """
    with tf.name_scope('sample_seq'):
        batch_size, *_ = utils.shape_list(context)
        
        # prefill. prompt context输出
        output = step(model_hparams, context)
        logits = tf.cast(output['logits'][:, -1], tf.float32)
        beta = 1 / tf.maximum(tf.cast(temperature, tf.float32), 1e-10)
        
        # 预测第一个token的logits. (512, 50257)
        first_output_logits = tf.cast(beta, logits.dtype) * logits
        print(f"first_output_logits.shape: {first_output_logits.shape}") # (512, 50257)

        # 预测第一个token的label. (512, )
        first_outputs = utils.sample_from_logits(first_output_logits)
        print(f"first_outputs.shape: {first_outputs.shape}") # (512,)

        # 预测第一个token的logprobs. (512, )
        first_logprobs = utils.logprobs_from_logits(
            logits=first_output_logits, 
            labels=first_outputs
        )
        print(f"first_logprobs.shape: {first_logprobs.shape}") # (512,)
        
        def _body(past, prev, output, logprobs, *extras):
            """
            循环体. 逐步生成token、更新模型状态, 并收集相关输出
            @past:      KVcache
            @prev:      上一次token
            @output:    当前为止生成序列
            @logprobs:  到目前所有生成token的logprobs
            @*extras:   其它额外要记录的信息(比如模型吐出的extra features)
            """
            next_outputs = step(
                model_hparams,              # model_hparams.
                prev[:, tf.newaxis],        # tokens. tf.expand_dims(prev, axis=1). (512,)->(512, 1)
                past=past,                  # past. (batch_size, n_layer, 2, n_head, sequence, head_dim)
                past_tokens=output[:, :-1]  # past_tokens. (batch_size, sequence). output[:, :-1]用于对齐past_tokens与past的长度
            )
            
            # logits: (512, 1, 50257)
            logits = tf.cast(next_outputs['logits'], tf.float32) * beta
            
            if top_k != 0:
                logits = tf.cond(
                    tf.equal(top_k, 0),
                    lambda: logits,
                    lambda: utils.take_top_k_logits(logits, top_k)
                )
                
            if top_p != 1.0:
                logits = utils.take_top_p_logits(logits, top_p)
                
            # next_sample: (512, 1)
            next_sample = utils.sample_from_logits(logits, dtype=tf.int32)
            
            # next_logprob: (512, 1)
            next_logprob = utils.logprobs_from_logits(
                logits=logits, 
                labels=next_sample
            )
            print(f"logits.shape: {logits.shape}, next_sample.shape: {next_sample.shape}, next_logprob: {next_logprob}")
            # logits.shape: (512, 1, 50257), next_sample.shape: (512, 1), next_logprob: Tensor("sample_seq/while/Neg:0", shape=(512, 1), dtype=float32)
            
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),   # past. present: [batch, layers, 2, heads, sequence, features]
                tf.squeeze(next_sample, axis=[1]),                      # prev. (512, 1) -> (512,)
                tf.concat([output, next_sample], axis=1),               # output. (512, seq) -> (512, seq+1)
                tf.concat([logprobs, next_logprob], axis=1),            # logprobs. (512, seq) -> (512, seq+1)
                *[tf.concat([prev, next_outputs[k]], axis=1) for k, prev in zip(extra_outputs, extras)], # extras. (512, seq)->(512, seq+1)
            ]

        try:
            shape_batch_size = int(batch_size)
        except TypeError:
            shape_batch_size = None
        
        if cond is None:
            def always_true(*args):
                return True
            cond = always_true
            
        """
        tf.while_loop()
        @body: 每步执行体
        @cond: 循环条件(如是否到最长/满足终止条件), 决定while_loop是否继续
        @loop_vars: 初始循环变量
        @shape_invariants: 每步循环变量的shape不变性约束
        @maximum_iterations: 最多循环多少步(如max_len-1)
        @back_prop: 禁用反向传播(通常推理/生成时不用反向)
        @parallel_iterations: 允许的并行循环步骤数, 提高吞吐量
        """
        _past, _, _output, _logprobs, *extras = tf.while_loop(
            body=_body,
            cond=cond,
            loop_vars=[
                output['presents'],                                         # past. (batch_size, n_layer, 2, n_head, sequence, head_dim)
                first_outputs,                                              # prev. (512,)
                tf.concat([context, first_outputs[:, tf.newaxis]], axis=1), # output. (512, 64)+(512, 1)=(512, 65)
                first_logprobs[:, tf.newaxis],                              # logprobs. (512, 1)
                *[output[k][:, -1:] for k in extra_outputs]                 # extras. value: (batch, 1)
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=model_hparams, batch_size=shape_batch_size)),
                tf.TensorShape([shape_batch_size]),
                tf.TensorShape([shape_batch_size, None]),
                tf.TensorShape([shape_batch_size, None]),
                *[tf.TensorShape([shape_batch_size, None]) for _ in extra_outputs]
            ],
            maximum_iterations=length-1, # 因为first是手动生成的, 后续只需要循环生成length-1
            back_prop=False,
            parallel_iterations=2,
        )
        
        return dict(
            tokens=_output,                     # tokens. (batch_size, context_length+sequence)
            presents=_past,                     # presents. (batch_size, n_layer, 2, n_head, context_length+sequence, head_dim)
            logprobs=_logprobs,                 # logprobs. (batch_size, sequence)
            **dict(zip(extra_outputs, extras))  # values. (batch_size, sequence)
        )
