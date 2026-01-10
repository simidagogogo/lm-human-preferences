#!/usr/bin/env python3

import os
from functools import partial

from mpi4py import MPI
import tensorflow as tf

from lm_human_preferences.utils import launch, hyperparams
from lm_human_preferences.utils import core as utils
from lm_human_preferences.policy import Policy
from lm_human_preferences.language import trained_models
from lm_human_preferences import lm_tasks
from lm_human_preferences import train_policy


def sample_policy(save_dir=None, savescope='policy', temperature=1.0, seed=None, batch_size=4, nsamples=0):
    """
    @save_dir:      模型参数和超参数保存的目录
    @savescope:     模型作用域命名(通常为policy)
    @temperature:   采样温度(采样多样性调节)
    @seed:          随机种子（可复现实验）
    @batch_size:    每轮采样多少条(query/response对)
    @nsamples:      总采样数量(0为无限采样)
    """
    hparams = train_policy.HParams()
    hparams.override_from_json_file(os.path.join(save_dir, 'train_policy_hparams.json'))
    hyperparams.dump(hparams)

    task = hparams.task
    comm = MPI.COMM_WORLD
    
    # 支持多进程采样. 每一张卡/进程负责nsamples_per_rank个样本
    nsamples_per_rank = utils.exact_div(nsamples, comm.Get_size())
    print(f"debug. task: {task}, nsamples: {nsamples}, comm.Get_size(): {comm.Get_size()}, nsamples_per_rank: {nsamples_per_rank}")
    
    with tf.Graph().as_default():
        m = trained_models.TrainedModel(
            name='sample', 
            savedir=os.path.join(save_dir, 'policy'), 
            scope='policy'
        )
        encoder = m.encoding.get_encoder()
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
                queries = sample_queries()
                rollouts = policy.respond(queries, length=task.response_length)
                assert len(queries.tolist()) == batch_size
                assert len(rollouts['responses'].tolist()) == batch_size
                for q, r in zip(queries.tolist(), rollouts['responses'].tolist()):
                    print('=' * 80)
                    print(encoder.decode(q).replace("\n", "⏎"))
                    print(encoder.decode(r).replace("\n", "⏎"))
                generated += batch_size


def launch_sample(mode='local', mpi=8, **kwargs):
    launch.launch('sample', partial(sample_policy, **kwargs), mode=mode, mpi=mpi)


if __name__ == '__main__':
    launch.main(dict(
        sample=launch_sample,
    ))


"""
./sample.py sample --save_dir gs://jeffwu-rcall/results/safety/lmhf-sent-69c5170-1909161359/ --mpi 8
"""
