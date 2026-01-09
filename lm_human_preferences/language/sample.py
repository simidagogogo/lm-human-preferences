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
    Inputs:
        step: A function which takes model hparams, a tokens Tensor, past, and
            returns a dictionary with 'logits' and 'presents', and any extra vars.
        context: Includes start tokens.
        extra_outputs: Map from extra output key to dtype
    Returns:
        A dict with keys 'presents', 'logits', and any keys in extra_outputs
    """
    with tf.name_scope('sample_seq'):
        batch_size, *_ = utils.shape_list(context)
        print(f"debug. batch_size: {batch_size}")
        context_output = step(model_hparams, context)
        
        print(f"context_output['logits'].shape: {context_output['logits'].shape}")
        # context_output['logits'].shape: (512, 64, 50257)
        
        logits = tf.cast(context_output['logits'][:, -1], tf.float32)
        beta = 1 / tf.maximum(tf.cast(temperature, tf.float32), 1e-10)
        first_output_logits = tf.cast(beta, logits.dtype) * logits
        print(f"first_output_logits.shape: {first_output_logits.shape}") # (512, 50257)

        first_outputs = utils.sample_from_logits(first_output_logits)
        print(f"first_outputs.shape: {first_outputs.shape}") # (512,)

        first_logprobs = utils.logprobs_from_logits(logits=first_output_logits, labels=first_outputs)
        print(f"first_logprobs.shape: {first_logprobs.shape}") # (512,)

        
        def body(past, prev, output, logprobs, *extras):
            """
            body函数通常作为循环体(例如tf.while_loop、tf.scan等)用于自回归文本生成(解码阶段), 即逐步生成token、更新模型状态, 并收集相关输出
            
            past: 历史模型缓存(注意力的KV缓存, 提升高效自回归解码)
            prev: 上一步生成的token
            output: 已生成的序列(到当前步为止)
            logprobs: 到目前为止所有生成token的log-probability
            *extras: 其它额外要记录的信息(比如模型吐出的extra features)
            """
            
            # 1.获取下一个token的logits
            next_outputs = step(
                model_hparams, 
                prev[:, tf.newaxis], 
                past=past,
                past_tokens=output[:, :-1]
            )
            
            # 2.应用采样温度(温度越高采样越发散)
            logits = tf.cast(next_outputs['logits'], tf.float32) * beta
            
            # 3.top-k筛选
            if top_k != 0:
                logits = tf.cond(
                    tf.equal(top_k, 0),
                    lambda: logits,
                    lambda: utils.take_top_k_logits(logits, top_k)
                )
                
            # 4.top-p(nucleus)筛选
            if top_p != 1.0:
                logits = utils.take_top_p_logits(logits, top_p)
                
            # 5.采样token
            # 按概率（经过top-k/top-p/温度变换后的logits）采样下一个 token 的 id。
            next_sample = utils.sample_from_logits(logits, dtype=tf.int32)

            # 6.计算 log probability
            # 计算本步采样token的负log概率(对后续分析, 例如ppl/beam search/EOS检测等有用)
            next_logprob = utils.logprobs_from_logits(logits=logits, labels=next_sample)
            print(f"logits.shape: {logits.shape}, next_sample.shape: {next_sample.shape}, next_logprob: {next_logprob}")
            # logits.shape: (512, 1, 50257), next_sample.shape: (512, 1), next_logprob: Tensor("sample_seq/while/Neg:0", shape=(512, 1), dtype=float32)
            
            # 7.拼接所有输出
            # 逐步把新生成结果融进历史, 确保每步循环完都可以继续往下生成. 支持自动记录额外特征(extras)
            
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),   # 新的 KV 记录
                tf.squeeze(next_sample, axis=[1]),                      # 当前步tokenid, 降维
                tf.concat([output, next_sample], axis=1),               # 当前步已生成序列
                tf.concat([logprobs, next_logprob], axis=1),            # 累积所有 token 的概率
                *[tf.concat([prev, next_outputs[k]], axis=1) 
                  for k, prev in zip(extra_outputs, extras)],           # 处理额外要输出的特征
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
        4. 整体循环作用
        整个 body 均以 batch 为单位，每步采样下一个 token，并串接过去历史：
            1. 更新 attention 缓存
            2. 记下所有已生成字，及生成过程概率
            3. 可以记录 extra features （比如 hidden states）
        整体可用于自回归生成循环，直到生成结束符（EOS）或达到最大长度。
        
        总结
        你可以把它理解为：文本生成的推进器
        每一步：
            1.预测下一个 token 分布
            2.按分布采样（可用 top-k/top-p、温度控制进行变换增强）
            3.记录序列，概率，模型内部 cache
            4.循环执行，最终得到完整生成序列和相关数据信息
        常见场景：GPT、Transformer 解码，ChatGPT 完整推理，从 prompt 开始每步生成、直到完成。
        """
        
        # 这是经典的自回归文本生成循环部分，配合 GPT/Transformer 使用，逐步采样下一个 token 并更新模型状态，直到达到指定长度。
        # 整体流程可视化
        # 1.初始化 loop_vars（先有prompt context的输出、第一步token等）
        # 2.按条件（cond）循环，每步调用 body：
        #   根据 past、prev、output、logprobs，输出下一个token及概率
        #   更新KV缓存、output序列、log概率等
        #   拼接或储存 extras
        # 3. 循环直到 maximum_iterations 或 cond终止条件达成
        # 4. 整体输出完整生成序列及各步相关信息
        """
        body: 每步执行体（即之前详细解释的 body 函数，一步生成）
        cond: 循环条件（如是否到最长/满足终止条件），决定 while_loop 是否继续
        loop_vars: 初始循环变量
        shape_invariants: 每步循环变量的 shape 不变性约束
        maximum_iterations: 最多循环多少步（如 max_len-1）
        back_prop: 禁用反向传播（通常推理/生成时不用反向）
        parallel_iterations: 允许的并行循环步骤数，提高吞吐量
        
        输出结果:
        presents: 最终的 KV 缓存（注意力past），可用于后续生成
        _: prev，最终都只是补位
        tokens: 生成的 token 序列（shape: [batch_size, out_length]）
        logprobs: 每个token的log概率（shape: [batch_size, out_length]）
        extras: 其它附加信息（如hidden states, attention maps等）
        """
        presents, _, tokens, logprobs, *extras = tf.while_loop(
            body=body,
            cond=cond,
            loop_vars=[
                context_output['presents'], # past
                first_outputs, # prev
                tf.concat([context, first_outputs[:, tf.newaxis]], axis=1), # output
                first_logprobs[:, tf.newaxis], #logprobs
                *[context_output[k][:, -1:] for k in extra_outputs] # extras
            ],
            
            # 这项规定每步循环中各变量的 shape 变化约束（TensorFlow 构建静态图的要求）。
            # 通常 shape_invariants 里用 None 表示不断扩展，如序列长度会增长。
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=model_hparams, batch_size=shape_batch_size)),
                tf.TensorShape([shape_batch_size]),
                tf.TensorShape([shape_batch_size, None]),
                tf.TensorShape([shape_batch_size, None]),
                *[tf.TensorShape([shape_batch_size, None]) for _ in extra_outputs]
            ],
            maximum_iterations=length-1,
            back_prop=False,
            parallel_iterations=2,
        )
        
        # logits：下一个 token 的 logits 分布。
        # presents：新一轮 KV 缓存（新的一步隐藏值）
        return dict(
            tokens=tokens, 
            presents=presents, 
            logprobs=logprobs, 
            **dict(zip(extra_outputs, extras))
        )
