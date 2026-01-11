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
    import json
    with open('/root/PycharmProjects/lm-human-preferences/.cursor/debug.log', 'a') as f:
        f.write(json.dumps(
            {'sessionId': 'debug-session', 'runId': 'run1', 'hypothesisId': 'A', 'location': 'sample.py:28', 'message': 'sample_policy entry', 
             'data': {'save_dir': save_dir, 'model_name': model_name, 'savescope': savescope, 'temperature': temperature, 'seed': seed, 'batch_size': batch_size, 'nsamples': nsamples}, 
             'timestamp': int(__import__('time').time() * 1000)
            }
        ))
        f.write("\n")
    
    comm = MPI.COMM_WORLD
    
    # 支持两种模式: 使用训练后的策略模型, 或使用原始 GPT-2 模型
    if model_name is not None:
        # 模式2：使用原始 GPT-2 模型，使用默认配置
        # #region agent log
        with open('/root/PycharmProjects/lm-human-preferences/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({'sessionId': 'debug-session', 'runId': 'run1', 'hypothesisId': 'A', 'location': 'sample.py:45', 'message': 'using raw GPT-2 model', 'data': {'model_name': model_name}, 'timestamp': int(__import__('time').time() * 1000)}) + '\n')
        # #endregion
        
        # 创建默认任务配置(基于launch.py中的books_task)
        hparams = train_policy.HParams()
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
        with open('/root/PycharmProjects/lm-human-preferences/.cursor/debug.log', 'a') as f:
            f.write(json.dumps(
                {'sessionId': 'debug-session', 'runId': 'run1', 'hypothesisId': 'A', 'location': 'sample.py:72', 'message': 'using trained policy model', 
                 'data': {'save_dir': save_dir}, 'timestamp': int(__import__('time').time() * 1000)}
            ))
            f.write("\n")
        
        hparams = train_policy.HParams()
        hparams_file = os.path.join(save_dir, 'train_policy_hparams.json')

        # region agent log
        with open('/root/PycharmProjects/lm-human-preferences/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({'sessionId': 'debug-session', 'runId': 'run1', 'hypothesisId': 'B', 'location': 'sample.py:78', 'message': 'before override_from_json_file', 'data': {'hparams_file': hparams_file, 'file_exists': os.path.exists(hparams_file)}, 'timestamp': int(__import__('time').time() * 1000)}) + '\n')
        # endregion

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
            # 变量初始化
            init_ops.run()
            
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

pipenv run ./sample.py sample --model_name 124M --mpi 1 --batch_size 2 --nsamples 4

pipenv run ./sample.py sample --save_dir /path/to/trained/policy --mpi 1
"""
