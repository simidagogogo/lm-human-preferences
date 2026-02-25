#!/usr/bin/env python3

import os
from functools import partial

# Try to import mpi4py, if it fails, use our mock
try:
    from mpi4py import MPI
except ImportError:
    import sys
    import os
    # Add current directory to path and import mock
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mpi_mock
    sys.modules['mpi4py'] = mpi_mock
    sys.modules['mpi4py.MPI'] = mpi_mock.MPI
    from mpi_mock import MPI

import tensorflow as tf

from lm_human_preferences.utils import launch, hyperparams
from lm_human_preferences.utils import core as utils
from lm_human_preferences.policy import Policy
from lm_human_preferences.language import trained_models
from lm_human_preferences import lm_tasks
from lm_human_preferences import train_policy


def sample_policy(save_dir=None, model_name=None, savescope='policy', temperature=1.0, seed=None, batch_size=4, nsamples=0):
    """
    @save_dir:      模型参数和超参数保存的目录（用于加载训练后的策略模型）
    @model_name:    原始 GPT-2 模型名称（如 '124M'），用于直接使用原始模型采样
    @savescope:     模型作用域命名(通常为policy)
    @temperature:   采样温度(采样多样性调节)
    @seed:          随机种子（可复现实验）
    @batch_size:    每轮采样多少条(query/response对)
    @nsamples:      总采样数量(0为无限采样)
    
    注意: save_dir 和 model_name 二选一。如果提供 model_name，则使用原始 GPT-2 模型和默认配置。
    """
    comm = MPI.COMM_WORLD
    # 支持两种模式: 使用训练后的策略模型, 或使用原始 GPT-2 模型
    if model_name is not None:
        # 模式2：使用原始 GPT-2 模型，使用默认配置
        # 创建默认任务配置(基于launch.py中的books_task)
        task = lm_tasks.TaskHParams(
            query_length=64,
            query_dataset='books',
            query_prefix='',
            query_suffix='',
            start_text='.',
            end_text='.',
            response_length=24,
            truncate_token=13,
            truncate_after=16,
            penalty_reward_value=-1,
        )
        task.policy.temperature = temperature
        task.policy.initial_model = model_name
        hparams = train_policy.HParams()
        hparams.task = task
        
        """
        ==========================
        hparams:
            ppo:
                batch_size: 64
                cliprange: 0.2
                cliprange_value: 0.2
                gamma: 1
                lam: 0.95
                lr: 5e-06
                nminibatches: 1
                noptepochs: 4
                total_episodes: 2000000
                vf_coef: 0.1
                whiten_rewards: True
            rewards:
                adaptive_kl: None
                kl_coef: 0.2
                train_new_model: None
                trained_model: None
            run:
                log_interval: 10
                save_dir: None
                save_interval: 50
                seed: None
            task:
                end_text: .
                penalty_reward_value: -1
                policy:
                    initial_model: 124M
                    temperature: 1.0
                query_dataset: books
                query_length: 64
                query_prefix: 
                query_suffix: 
                response_length: 24
                start_text: .
                truncate_after: 16
                truncate_token: 13
        ==========================
        """
        hyperparams.dump(hparams)
        
        # 使用原始 GPT-2 模型（当 savedir=None 时，TrainedModel 会从 GPT2_MODEL_PATH/models/name 加载）
        # 原始 GPT-2 checkpoint 的变量名是 model/...，所以不需要 scope 前缀
        model_savedir = None
        model_name_for_loading = model_name
        model_scope = None  # 原始模型不需要 scope 前缀
    elif save_dir is not None:
        # 模式1：使用训练后的策略模型
        hparams = train_policy.HParams()
        hparams_file = os.path.join(save_dir, 'train_policy_hparams.json')
        hparams.override_from_json_file(hparams_file)
        hyperparams.dump(hparams)
        task = hparams.task
        model_savedir = os.path.join(save_dir, 'policy')
        model_name_for_loading = 'sample'
        model_scope = 'policy'  # 训练后的模型使用 policy scope
    else:
        raise ValueError("必须提供 save_dir 或 model_name 参数之一")
    
    print(f"debug. model_savedir: {model_savedir}, model_name_for_loading: {model_name_for_loading}, model_scope: {model_scope}")
    # debug. model_savedir: None, model_name_for_loading: 124M, model_scope: None

    # 支持多进程采样. 每一张卡/进程负责nsamples_per_rank个样本
    nsamples_per_rank = utils.exact_div(nsamples, comm.Get_size())
    print(f"debug. task: {task}, nsamples: {nsamples}, comm.Get_size(): {comm.Get_size()}, nsamples_per_rank: {nsamples_per_rank}")
    """
    debug. task: TaskHParams(
        query_length=64, 
        query_dataset='books', 
        query_prefix='', 
        query_suffix='', 
        start_text='.', 
        end_text='.', 
        response_length=24, 
        truncate_token=13, 
        truncate_after=16, 
        penalty_reward_value=-1, 
        policy=PolicyHParams(temperature=1.0, initial_model='124M')
    ), 
    nsamples: 4, 
    comm.Get_size(): 1, 
    nsamples_per_rank: 4
    """

    with tf.Graph().as_default():
        m = trained_models.TrainedModel(
            name=model_name_for_loading, 
            savedir=model_savedir, 
            scope=model_scope
        )

        # 可逆分词器与编码器(ReversibleEncoder类的实例)
        encoder = m.encoding.get_encoder()
        
        """
        ==========================
        model_hparams:
            attn_pdrop: 0.1
            embd_pdrop: 0.1
            head_pdrop: 0.1
            n_ctx: 1024
            n_embd: 768
            n_head: 12
            n_layer: 12
            n_vocab: 50257
            resid_pdrop: 0.1
        ==========================
        """
        hyperparams.dump(m.hparams(), name='model_hparams')
        utils.set_mpi_seed(seed)
        policy = Policy(
            m, 
            scope='policy',
            is_root=True, # just init on every rank, simplifies code
            embed_queries=lm_tasks.query_formatter(task, encoder),
            temperature=temperature,
        )

        # 负责批量采集queries(prompt/query生成器)
        query_sampler = lm_tasks.make_query_sampler(
            hparams=task, 
            encoder=encoder, 
            comm=comm,
            batch_size=batch_size, 
            mode='test'
        )

        # TF1风格, 需手动变量初始化
        init_ops = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        )

        # 创建TF会话
        with utils.mpi_session() as sess:
            init_ops.run()  # 变量初始化
            
            @utils.graph_function()
            def sample_queries():
                return query_sampler()['tokens']
            tf.get_default_graph().finalize()

            """
            主循环直到采集/生成符合数量样本
            采样一组queries, 调用policy的respond批量生成responses(rollouts), 并逐对显示query和response
            每次批处理递增采样数
            """
            generated = 0
            while nsamples_per_rank == 0 or generated < nsamples_per_rank:
                print(f"generated: {generated}")
                queries = sample_queries()
                rollouts = policy.respond(queries, length=task.response_length)
                assert len(queries.tolist()) == batch_size
                assert len(rollouts['responses'].tolist()) == batch_size
                for q, r in zip(queries.tolist(), rollouts['responses'].tolist()):
                    print('=' * 80)
                    que = encoder.decode(q).replace("\n", "⏎")
                    res = encoder.decode(r).replace("\n", "⏎")
                    print(f"que: {que}")
                    print(f"res: {res}")
                generated += batch_size


def launch_sample(mode='local', mpi=8, **kwargs):
    launch.launch('sample', partial(sample_policy, **kwargs), mode=mode, mpi=mpi)


if __name__ == '__main__':
    launch.main(dict(
        sample=launch_sample,
    ))


"""
./sample.py sample --save_dir gs://jeffwu-rcall/results/safety/lmhf-sent-69c5170-1909161359/ --mpi 8
pipenv run ./sample.py sample --save_dir /path/to/trained/policy --mpi 1
"""

"""
pipenv run ./sample.py sample --model_name 124M --mpi 1 --batch_size 2 --nsamples 4
(base) root@iZ0jlfyn5du7ptefx2tr5vZ:~/PycharmProjects/lm-human-preferences# pipenv run ./sample.py sample --model_name 124M --mpi 1 --batch_size 2 --nsamples 4
name: main, self.base_path: /root/gpt-2-models/encodings/main
name: main, self.encoder_path: /root/gpt-2-models/encodings/main/encoder.json
name: main, self.bpe_path: /root/gpt-2-models/encodings/main/vocab.bpe
name: test, self.base_path: gs://gpt-2/encodings/test
debug. enter launch(). name: sample, f: functools.partial(<function sample_policy at 0x7f6e2d243290>, model_name='124M', batch_size=2, nsamples=4)
==========================
hparams:
    ppo:
        batch_size: 64
        cliprange: 0.2
        cliprange_value: 0.2
        gamma: 1
        lam: 0.95
        lr: 5e-06
        nminibatches: 1
        noptepochs: 4
        total_episodes: 2000000
        vf_coef: 0.1
        whiten_rewards: True
    rewards:
        adaptive_kl: None
        kl_coef: 0.2
        train_new_model: None
        trained_model: None
    run:
        log_interval: 10
        save_dir: None
        save_interval: 50
        seed: None
    task:
        end_text: .
        penalty_reward_value: -1
        policy:
            initial_model: 124M
            temperature: 1.0
        query_dataset: books
        query_length: 64
        query_prefix: 
        query_suffix: 
        response_length: 24
        start_text: .
        truncate_after: 16
        truncate_token: 13
==========================
debug. model_savedir: None, model_name_for_loading: 124M, model_scope: None
debug. task: TaskHParams(query_length=64, query_dataset='books', query_prefix='', query_suffix='', start_text='.', end_text='.', response_length=24, truncate_token=13, truncate_after=16, penalty_reward_value=-1, policy=PolicyHParams(initial_model='124M', temperature=1.0)), nsamples: 4, comm.Get_size(): 1, nsamples_per_rank: 4
local_model_path: /root/gpt-2-models/models/124M
name: 124M, scope: None, self.savedir: /root/gpt-2-models/models/124M, self.encoding: <lm_human_preferences.language.encodings.Encoding object at 0x7f6e0fe83790>
[hparams]. name: 124M, _hparams: HParams(n_vocab=50257, n_ctx=1024, n_embd=768, n_head=12, n_layer=12, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1, head_pdrop=0.1)
==========================
model_hparams:
    attn_pdrop: 0.1
    embd_pdrop: 0.1
    head_pdrop: 0.1
    n_ctx: 1024
    n_embd: 768
    n_head: 12
    n_layer: 12
    n_vocab: 50257
    resid_pdrop: 0.1
==========================
[hparams]. name: 124M, _hparams: HParams(n_vocab=50257, n_ctx=1024, n_embd=768, n_head=12, n_layer=12, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1, head_pdrop=0.1)
tiled_prefix.shape: [None, 0]
tiled_suffix.shape: [None, 0]
[respond_op]. contexts.shape: (?, ?), context_length: Tensor("strided_slice_1:0", shape=(), dtype=int32)
WARNING:tensorflow:From /root/.local/share/virtualenvs/lm-human-preferences-XpxZn-hG/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From /root/.local/share/virtualenvs/lm-human-preferences-XpxZn-hG/lib/python3.7/site-packages/tensorflow/python/framework/function.py:1007: calling Graph.create_op (from tensorflow.python.framework.ops) with compute_shapes is deprecated and will be removed in a future version.
Instructions for updating:
Shapes are always computed; don't use the compute_shapes as it has no effect.
[__call__]. results: {'present': <tf.Tensor 'sample_seq/step/policy/model/stack:0' shape=(?, 12, 2, 12, ?, 64) dtype=float32>, 'h': <tf.Tensor 'sample_seq/step/policy/model/index_each:0' shape=(?, ?, 768) dtype=float32>, 'lm_all_losses': <tf.Tensor 'sample_seq/step/policy/model/strided_slice_7:0' shape=(?, ?) dtype=float32>, 'lm_logits': <tf.Tensor 'sample_seq/step/policy/model/Reshape_3:0' shape=(?, ?, ?) dtype=float32>, 'lm_losses': <tf.Tensor 'sample_seq/step/policy/model/Mean:0' shape=(?,) dtype=float32>, 'value': <tf.Tensor 'sample_seq/step/policy/model/heads/value/add:0' shape=(?, ?) dtype=float32>, 'value_regularizer': <tf.Tensor 'sample_seq/step/policy/model/heads/value/l2_regularizer:0' shape=() dtype=float32>}
self.savedir: /root/gpt-2-models/models/124M
ckpt: /root/gpt-2-models/models/124M/model.ckpt
Param policy/model/heads/value/w is missing from checkpoint /root/gpt-2-models/models/124M/model.ckpt
Param policy/model/heads/value/b is missing from checkpoint /root/gpt-2-models/models/124M/model.ckpt
[step_core]. logits.shape: (?, ?, ?), presents.shape: (?, 12, 2, 12, ?, 64), value.shape: (?, ?)
first_output_logits.shape: (?, ?)
first_outputs.shape: (?,)
first_logprobs.shape: (?,)
[__call__]. results: {'present': <tf.Tensor 'sample_seq/while/step/stack:0' shape=(?, 12, 2, 12, ?, 64) dtype=float32>, 'h': <tf.Tensor 'sample_seq/while/step/index_each:0' shape=(?, 1, 768) dtype=float32>, 'lm_all_losses': <tf.Tensor 'sample_seq/while/step/strided_slice_7:0' shape=(?, 0) dtype=float32>, 'lm_logits': <tf.Tensor 'sample_seq/while/step/Reshape_3:0' shape=(?, 1, ?) dtype=float32>, 'lm_losses': <tf.Tensor 'sample_seq/while/step/Mean:0' shape=(?,) dtype=float32>, 'value': <tf.Tensor 'sample_seq/while/step/heads/value/add:0' shape=(?, 1) dtype=float32>, 'value_regularizer': <tf.Tensor 'sample_seq/while/step/heads/value/l2_regularizer:0' shape=() dtype=float32>}
[step_core]. logits.shape: (?, 1, ?), presents.shape: (?, 12, 2, 12, ?, 64), value.shape: (?, 1)
logits.shape: (?, 1, ?), next_sample.shape: (?, 1), next_logprob: Tensor("sample_seq/while/Neg:0", shape=(?, 1), dtype=float32)
tiled_prefix.shape: [None, 0]
tiled_suffix.shape: [None, 0]
[__call__]. results: {'present': <tf.Tensor 'step/stack:0' shape=(?, 12, 2, 12, ?, 64) dtype=float32>, 'h': <tf.Tensor 'step/index_each:0' shape=(?, ?, 768) dtype=float32>, 'lm_all_losses': <tf.Tensor 'step/strided_slice_7:0' shape=(?, ?) dtype=float32>, 'lm_logits': <tf.Tensor 'step/Reshape_3:0' shape=(?, ?, ?) dtype=float32>, 'lm_losses': <tf.Tensor 'step/Mean:0' shape=(?,) dtype=float32>, 'value': <tf.Tensor 'step/heads/value/add:0' shape=(?, ?) dtype=float32>, 'value_regularizer': <tf.Tensor 'step/heads/value/l2_regularizer:0' shape=() dtype=float32>}
[step_core]. logits.shape: (?, ?, ?), presents.shape: (?, 12, 2, 12, ?, 64), value.shape: (?, ?)
[analyze_responses_op]. logits.shape: (?, ?, ?), values.shape: (?, ?)
[analyze_responses_op]. logprobs: Tensor("Neg:0", shape=(?, ?), dtype=float32), entropies: Tensor("sub_1:0", shape=(?, ?), dtype=float32)
debug. sequence_length: 64, mode: test, repeat_count: None
start_token: 13, end_token: 13, padding_token: None
padding_token: 50259
WARNING:tensorflow:From /root/.local/share/virtualenvs/lm-human-preferences-XpxZn-hG/lib/python3.7/site-packages/tensorflow/python/data/ops/dataset_ops.py:429: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
Instructions for updating:
tf.py_func is deprecated in TF V2. Instead, use
    tf.py_function, which takes a python function which manipulates tf eager
    tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
    an ndarray (just call tensor.numpy()) but having access to eager tensors
    means `tf.py_function`s can use accelerators such as GPUs as well as
    being differentiable using a gradient tape.
    
2026-02-10 21:26:11.199441: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2026-02-10 21:26:11.233774: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2699995000 Hz
2026-02-10 21:26:11.234023: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x3b8f81b0 executing computations on platform Host. Devices:
2026-02-10 21:26:11.234049: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
generated: 0
WARNING: Dataset file appears to be empty or placeholder. Using test data generator.
================================================================================
tokens: [304, 474, 304, 331, 288, 256, 1976, 1312, 374, 266, 1976, 256, 304, 13, 288, 2124, 269, 410, 479, 279, 374, 288, 300, 13, 479, 256, 334, 308, 374, 279, 267, 10662, 1312, 275, 1976, 13, 257, 269, 2124, 285, 266, 1976, 410, 334, 257, 256, 279, 479, 289, 13, 266, 269, 308, 264, 289, 289, 1976, 304, 1976, 374, 13, 50259, 50259, 50259]
unicode text: ĠeĠjĠeĠyĠdĠtĠzĠiĠrĠwĠzĠtĠe.ĠdĠxĠcĠvĠkĠpĠrĠdĠl.ĠkĠtĠuĠgĠrĠpĠoĠqĠiĠbĠz.ĠaĠcĠxĠmĠwĠzĠvĠuĠaĠtĠpĠkĠh.ĠwĠcĠgĠsĠhĠhĠzĠeĠzĠr.
final text:  e j e y d t z i r w z t e. d x c v k p r d l. k t u g r p o q i b z. a c x m w z v u a t p k h. w c g s h h z e z r.
tokens: [277, 374, 1312, 256, 256, 299, 269, 479, 304, 300, 304, 275, 304, 1976, 13, 289, 277, 479, 257, 256, 304, 264, 1312, 267]
unicode text: ĠfĠrĠiĠtĠtĠnĠcĠkĠeĠlĠeĠbĠeĠz.ĠhĠfĠkĠaĠtĠeĠsĠiĠo
final text:  f r i t t n c k e l e b e z. h f k a t e s i o
que:  e j e y d t z i r w z t e. d x c v k p r d l. k t u g r p o q i b z. a c x m w z v u a t p k h. w c g s h h z e z r.
res:  f r i t t n c k e l e b e z. h f k a t e s i o
================================================================================
tokens: [474, 266, 288, 374, 479, 374, 308, 1976, 256, 374, 264, 474, 267, 13, 256, 1976, 285, 479, 264, 289, 13, 277, 308, 277, 275, 256, 410, 1312, 279, 269, 13, 410, 331, 304, 304, 275, 269, 13, 410, 285, 266, 10662, 1312, 10662, 1976, 289, 308, 410, 264, 299, 264, 13, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259]
unicode text: ĠjĠwĠdĠrĠkĠrĠgĠzĠtĠrĠsĠjĠo.ĠtĠzĠmĠkĠsĠh.ĠfĠgĠfĠbĠtĠvĠiĠpĠc.ĠvĠyĠeĠeĠbĠc.ĠvĠmĠwĠqĠiĠqĠzĠhĠgĠvĠsĠnĠs.
final text:  j w d r k r g z t r s j o. t z m k s h. f g f b t v i p c. v y e e b c. v m w q i q z h g v s n s.
tokens: [304, 376, 308, 311, 2808, 13, 352, 267, 374, 266, 300, 1511, 334, 1312, 266, 374, 1312, 285, 267, 13, 264, 285, 257, 299]
unicode text: ĠeĠFĠgĠSĠ29.Ġ1ĠoĠrĠwĠlĠ13ĠuĠiĠwĠrĠiĠmĠo.ĠsĠmĠaĠn
final text:  e F g S 29. 1 o r w l 13 u i w r i m o. s m a n
que:  j w d r k r g z t r s j o. t z m k s h. f g f b t v i p c. v y e e b c. v m w q i q z h g v s n s.
res:  e F g S 29. 1 o r w l 13 u i w r i m o. s m a n
generated: 2
================================================================================
tokens: [479, 308, 289, 257, 2124, 1312, 288, 266, 289, 300, 1976, 277, 479, 299, 275, 13, 1976, 304, 266, 289, 275, 264, 13, 374, 256, 410, 269, 257, 288, 334, 308, 256, 264, 288, 285, 269, 300, 288, 13, 256, 257, 308, 277, 266, 13, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259]
unicode text: ĠkĠgĠhĠaĠxĠiĠdĠwĠhĠlĠzĠfĠkĠnĠb.ĠzĠeĠwĠhĠbĠs.ĠrĠtĠvĠcĠaĠdĠuĠgĠtĠsĠdĠmĠcĠlĠd.ĠtĠaĠgĠfĠw.
final text:  k g h a x i d w h l z f k n b. z e w h b s. r t v c a d u g t s d m c l d. t a g f w.
tokens: [267, 374, 308, 304, 299, 264, 257, 266, 285, 1312, 299, 764, 374, 264, 256, 334, 288, 299, 267, 374, 256, 304, 5145, 502]
unicode text: ĠoĠrĠgĠeĠnĠsĠaĠwĠmĠiĠnĠ.ĠrĠsĠtĠuĠdĠnĠoĠrĠtĠeĠ!Ġme
final text:  o r g e n s a w m i n . r s t u d n o r t e ! me
que:  k g h a x i d w h l z f k n b. z e w h b s. r t v c a d u g t s d m c l d. t a g f w.
res:  o r g e n s a w m i n . r s t u d n o r t e ! me
================================================================================
tokens: [269, 334, 474, 300, 299, 277, 275, 10662, 13, 275, 256, 288, 266, 285, 308, 1312, 300, 2124, 279, 264, 277, 13, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259]
unicode text: ĠcĠuĠjĠlĠnĠfĠbĠq.ĠbĠtĠdĠwĠmĠgĠiĠlĠxĠpĠsĠf.
final text:  c u j l n f b q. b t d w m g i l x p s f.
tokens: [308, 267, 334, 374, 331, 334, 334, 764, 300, 331, 362, 289, 513, 299, 604, 1058, 264, 304, 285, 1931, 299, 308, 334, 308]
unicode text: ĠgĠoĠuĠrĠyĠuĠuĠ.ĠlĠyĠ2ĠhĠ3ĠnĠ4Ġ:ĠsĠeĠmĠerĠnĠgĠuĠg
final text:  g o u r y u u . l y 2 h 3 n 4 : s e m er n g u g
que:  c u j l n f b q. b t d w m g i l x p s f.
res:  g o u r y u u . l y 2 h 3 n 4 : s e m er n g u g
"""
