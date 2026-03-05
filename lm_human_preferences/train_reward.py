#!/usr/bin/env python3

"""
奖励模型训练逻辑
"""

import json
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np
import tensorflow as tf
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
from tensorflow.contrib import summary

from lm_human_preferences import label_types, lm_tasks, rewards
from lm_human_preferences.language import trained_models
from lm_human_preferences.policy import Policy
from lm_human_preferences.utils import core as utils
from lm_human_preferences.utils import gcs, hyperparams
from lm_human_preferences.utils.core import Schema


@dataclass
class LabelHParams(hyperparams.HParams):
    type: str = None        # best_of_4
    num_train: int = None   # 样本条数, 形如4_992
    source: str = None      # 数据源, 形如'https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/sentiment/offline_5k.json'


@dataclass
class RunHParams(hyperparams.HParams):
    seed: Optional[int] = None      # 随机种子
    log_interval: int = 10          # 日志频率
    save_interval: int = 50         # 保存频率
    save_dir: Optional[str] = None  # 保存目录


@dataclass
class HParams(hyperparams.HParams):
    run: RunHParams = field(default_factory=RunHParams)
    task: lm_tasks.TaskHParams = field(default_factory=lm_tasks.TaskHParams)
    labels: LabelHParams = field(default_factory=LabelHParams)

    batch_size: int = 40            # total across ranks
    lr: float = 5e-5                # 学习率
    rollout_batch_size: int = 64    # 一次rollout包括64条样本
    normalize_samples: int = 0      # Samples used to estimate reward mean and std
    debug_normalize: int = 0        # Samples used to check that normalization worked

    # Whether, before training, to normalize the rewards on the policy to the scales on the training buffer.
    # (For comparisons, just use mean 0, var 1.)
    # 训练开始前, 把当前policy rollout的reward对齐到训练buffer的reward尺度, 主要为了训练稳定/公平
    normalize_before: bool = False
    
    # Whether, after training, to normalize the rewards on the ref policy to mean 0, var 1
    # (so the KL coefficient always has the same meaning).
    # 训练结束后, 把ref policy的reward标准化到N(0,1)这种固定尺度, 主要为了让KL系数β的含义稳定、不同实验可比
    normalize_after: bool = False

    def validate(self, *, prefix=''):
        super().validate(prefix=prefix)
        utils.exact_div(self.labels.num_train, self.batch_size)


def round_down_to_multiple(n, divisor):
    """
    把整数n向下取整到divisor的整数倍. 让序列长度/批大小对齐到固定块大小
    """
    return n - n % divisor


def download_labels(source, label_type, question_schemas, total_labels, comm):
    """从url中下载人类标注的数据集, 用于训练reward模型
    
    :param source: https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/descriptiveness/offline_5k.json
    :param label_type: 人类标注样本类型, label.type, 形如best_of_4
    :param total_labels: 最大样本条数, label.num_train, 形如4_992
    :param question_schemas: 
    :param comm: 分布式通信
    :return: 
        dict{
            'query': List[],
            'sample0': List[],
            'sample1': List[],
            ...
        }
    """
    
    """
    schemas: {
    'query':   Schema(dtype=tf.int32, shape=(64,)), 
    'sample0': Schema(dtype=tf.int32, shape=(24,)), 
    'sample1': Schema(dtype=tf.int32, shape=(24,)), 
    'sample2': Schema(dtype=tf.int32, shape=(24,)), 
    'sample3': Schema(dtype=tf.int32, shape=(24,)), 
    'best':    Schema(dtype=tf.int32, shape=())},
    """
    schemas = {
        **question_schemas, 
        **label_type.label_schemas()
    }
    """
    if self.is_root:
        with tf.device('cpu:0'):
            self._enqueue_phs = {
                name: tf.placeholder(name=name, dtype=schema.dtype, shape=(None,) + schema.shape)
                for name, schema in self.schemas.items()
            }
            self._enqueue_answers = self.answer_queue.enqueue_many(self._enqueue_phs)
    else:
        self._enqueue_phs = None
        self._enqueue_answers = None
    """

    # TODO: download on just one rank?  then do: labels = utils.mpi_bcast_tensor_dict(labels, comm=comm)
    if source != 'test':
        # 数据来自本地缓存
        with open(gcs.download_file_cached(source, comm=comm)) as f:
            results = json.load(f)
            print('Num labels found in source:', len(results)) # Num labels found in source: 6260
    else:
        results = [
            {
                name: np.zeros(schema.shape, dtype=schema.dtype.as_numpy_dtype)
                for name, schema in schemas.items()
            }
            for _ in range(50)
        ]

    assert len(results) >= total_labels
    results = results[:total_labels]
    # 类似于列变行
    return {
        k: [item[k] for item in results] 
        for k in schemas.keys()
    }


class RewardModelTrainer():
    """
    Reward Model训练器/训练流程管理器
    """
    def __init__(self, *, reward_model, policy, query_sampler, hparams, comm):
        """
        @reward_model:  用于计算奖励得分
        @policy:        ref_policy, 用于自回归生成文本
        @query_sampler: 从Dataset中采样prompts
        """
        self.reward_model = reward_model    # 待训练的奖励模型
        self.policy = policy                # ref_policy, reward模型训练过程中参数不会变化
        self.hparams = hparams
        self.num_ranks = comm.Get_size()
        self.rank = comm.Get_rank()
        self.comm = comm

        self.label_type = label_types.get(hparams.labels.type)
        self.question_schemas = self.label_type.question_schemas(
            query_length=hparams.task.query_length,
            response_length=hparams.task.response_length,
        )
        data_schemas = {
            **self.label_type.label_schemas(),
            **self.question_schemas,
        }
        """
        data_schemas: {
            'best':    Schema(dtype=tf.int32, shape=())},
            'query':   Schema(dtype=tf.int32, shape=(64,)), 
            'sample0': Schema(dtype=tf.int32, shape=(24,)), 
            'sample1': Schema(dtype=tf.int32, shape=(24,)), 
            'sample2': Schema(dtype=tf.int32, shape=(24,)), 
            'sample3': Schema(dtype=tf.int32, shape=(24,)), 
        """

        with tf.device(None), tf.device('/cpu:0'):
            with tf.variable_scope('label_buffer', use_resource=True, initializer=tf.zeros_initializer):
                # 数据来源为
                self.train_buffer = utils.SampleBuffer(
                    capacity=hparams.labels.num_train, 
                    schemas=data_schemas
                )

        with tf.name_scope('train_reward'):
            # TensorBoard 记录
            summary_writer = utils.get_summary_writer(
                self.hparams.run.save_dir, 
                subdir='reward_model', 
                comm=comm
            )

            @utils.graph_function(indices=Schema(tf.int32, (None,)), lr=Schema(tf.float32, ()))
            def train_batch(indices, lr):
                """定义单步训练函数(Training Step Function)
                在一个深度学习训练循环中，这相当于执行了一次 sess.run(train_step)
                它封装了数据读取、损失计算、反向传播（梯度下降）以及日志记录的全过程
                
                :param indices: 每个rank的训练样本, per_rank_batch_size=batch_size/num_ranks
                :parm lr: 学习率
                :return: 
                """
                with tf.name_scope('minibatch'):
                    # 1. 从经验池中随机采样indices条训练样本
                    minibatch = self.train_buffer.read(indices)
                    
                    # 2. 损失计算与优化
                    stats = self.label_type.loss(
                        reward_model=self.reward_model.get_rewards_op, 
                        labels=minibatch
                    )

                    # 3. 更新reward_model权重. 
                    # train_op是个操作节点. 只要运行它, 模型参数就会被更新一次
                    train_op = utils.minimize(
                        loss=stats['loss'], 
                        lr=lr, 
                        params=self.reward_model.get_params(), 
                        name='opt', 
                        comm=self.comm
                    )

                    # 4.训练步数与日志记录(Step & Logging)
                    with tf.control_dependencies([train_op]):
                        # step_var全局计数器变量 train_step
                        step_var = tf.get_variable(
                            name='train_step', 
                            dtype=tf.int64, 
                            shape=(), 
                            trainable=False, 
                            use_resource=True
                        )
                        step = step_var.assign_add(1) - 1
                        stats = utils.FlatStats.from_dict(stats).map_flat(
                            partial(utils.mpi_allreduce_mean, comm=comm)
                        ).as_dict()
                        train_stat_op = utils.record_stats(
                            stats=stats, 
                            summary_writer=summary_writer, 
                            step=step, 
                            log_interval=hparams.run.log_interval, 
                            comm=comm
                        )
                """
                依赖链: 
                    train_stat_op 依赖于日志记录操作。
                    日志记录操作依赖于train_op(由 control_dependencies保证)
                    train_op 依赖于loss 计算
                    loss 计算依赖于minibatch读取
                结论: 在Session中运行 sess.run(train_stat_op), 会触发整条链式反应, 完成一次完整的训练迭代
                """
                return train_stat_op
            self.train_batch = train_batch

        if self.hparams.normalize_before or self.hparams.normalize_after:
            @utils.graph_function()
            def target_mean_std():
                """
                Returns the means and variances to target for each reward model
                """
                # Should be the same on all ranks because the train_buf should be the same
                scales = self.label_type.target_scales(self.train_buffer.data())
                if scales is None:
                    return tf.zeros([]), tf.ones([])
                else:
                    mean, var = tf.nn.moments(scales, axes=[0])
                    return mean, tf.sqrt(var)
            self.target_mean_std = target_mean_std

            def stats(query_responses):
                """
                这是一个计算奖励模型在多个样本上的统计量（均值和标准差）的方法，用于 RLHF 训练中的奖励归一化。它使用 MPI（多进程通信） 在分布式环境中聚合统计信息。
                @query_responses: List[Tuple[querys, responses]], query的shape为[rollout_batch_size, xx]
                """
                # (normalize_samples, )
                rewards = np.concatenate(
                    [self.reward_model.get_rewards(qs, rs) for qs, rs in query_responses], 
                    axis=0
                )
                assert len(rewards.shape) == 1, f'{rewards.shape}'
                sums = np.asarray([rewards.sum(axis=0), np.square(rewards).sum(axis=0)])

                # TODO: allreduce
                means, sqr_means = self.comm.allreduce(sums, op=MPI.SUM) / (self.num_ranks * rewards.shape[0])
                stds = np.sqrt(sqr_means - means ** 2)
                return means, stds
            self.stats = stats

            def log_stats_after_normalize(stats):
                """log stats after normalize.
                """
                if comm.Get_rank() != 0:
                    return
                means, stds = stats
                print(f'after normalize: {means} +- {stds}')
            self.log_stats_after_normalize = log_stats_after_normalize

            def reset_reward_scales():
                """
                reward_model奖励归一化参数重置
                通过可控的缩放和平移参数来调整奖励信号的尺度, 这在强化学习训练中对稳定性和收敛速度非常重要
                """
                self.reward_model.reset_reward_scale()
            self.reset_reward_scales = reset_reward_scales

            def set_reward_norms(mean, std, new_mean, new_std):
                """
                调整reward_model奖励值归一化参数
                通过可控的缩放和平移参数来调整奖励信号的尺度, 这在强化学习训练中对稳定性和收敛速度非常重要
                """
                print(f'before normalize: {mean} +- {std}')
                print(f'targets: {new_mean} +- {new_std}')

                # 检查数组中所有元素是否都为有限值(只要有一个不为有限值就抛出异常)
                assert np.isfinite((mean, std, new_mean, new_std)).all()
                self.reward_model.set_reward_norm(
                    old_mean=mean, 
                    old_std=std, 
                    new_mean=new_mean, 
                    new_std=new_std
                )
            self.set_reward_norms = set_reward_norms

        # 训练前归一化 or 训练后归一化
        if self.hparams.normalize_before or self.hparams.normalize_after:
            @utils.graph_function()
            def sample_policy_batch():
                """
                将queries输入ref_policy模型并自回归得到responses
                @return: 返回一批(prompt, response)对，用于：
                    1. 计算reward分数
                    2. 统计reward分布
                    3. 调整归一化参数
                """
                # queries: [rollout_batch_size/comm.Get_size() , query_length]
                queries = query_sampler('ref_queries')['tokens']
                
                # responses: [rollout_batch_size, response_length]
                responses = policy.respond_op(
                    queries=queries, 
                    length=hparams.task.response_length
                )['responses']
                return queries, responses

            def sample_policy_responses(n_samples):
                """
                采样<query, response>n_samples对
                @n_samples: 一共采样的[queries, responses]条数
                @return: List[Tuplep[queries, responses]], list长度为n_batches, 其中每个queries的shape: [hparams.rollout_batch_size, xx]
                """
                n_batches = utils.ceil_div(n_samples, hparams.rollout_batch_size)
                return [sample_policy_batch() for _ in range(n_batches)]
            self.sample_policy_responses = sample_policy_responses

        @utils.graph_function(labels=utils.add_batch_dim(data_schemas))
        def add_to_buffer(labels):
            """将训练样本添加到buffer"""
            return self.train_buffer.add(**labels)
        self.add_to_buffer = add_to_buffer

    def normalize(self, sample_fn, target_means, target_stds):
        """
        归一化, 通过调整reward_model内部的gain和bias实现
        @sample_fn:     从ref_policy中采样<query, response>样本对的方法
        @target_means:  目标均值
        @target_stds:   目标方差
        @return: 内部会改变gain和bias
        """
        if not self.hparams.normalize_samples:
            return

        # step1. 重置reward_model的奖励归一化参数
        self.reset_reward_scales()
        
        # step2. 从策略样本采样数据以估计统计量
        query_responses = sample_fn(self.hparams.normalize_samples) # 256, len(query_responses)=256/64=4
        means, stds = self.stats(query_responses)

        # step3. 调整归一化参数gain/bias, 将当前分布 N(means, stds) 变换为 N(target_mean, target_std)
        self.set_reward_norms(means, stds, target_means, target_stds)
        if self.hparams.debug_normalize:
            query_responses = sample_fn(self.hparams.debug_normalize)
            stats = self.stats(query_responses)
            self.log_stats_after_normalize(stats)

    def train(self):
        """
        标准训练流程
        """
        # 1. 下载标签数据(人类偏好标注), 并添加到缓存区
        # labels = dict{
        #     'query': List[],
        #     'sample0': List[],
        #     'sample1': List[],
        #     ...
        # }
        labels = download_labels(
            self.hparams.labels.source,
            label_type=self.label_type,
            question_schemas=self.question_schemas,
            total_labels=self.hparams.labels.num_train,
            comm=self.comm
        )
        self.add_to_buffer(labels)

        # 2. 训练前归一化normalize_before
        if self.hparams.normalize_before:
            target_mean, target_std = self.target_mean_std()
            self.normalize(self.sample_policy_responses, target_mean, target_std)

        # 3. Train on train_indices
        # Collect training data for reward model training.  train_indices will include the indices
        # trained on across all ranks, and its size must be a multiple of minibatch_size.
        per_rank_batch_size = utils.exact_div(self.hparams.batch_size, self.num_ranks)
        # Make sure each rank gets the same shuffle so we train on each point exactly once
        train_indices = self.comm.bcast(np.random.permutation(self.hparams.labels.num_train))
        print(self.rank, " training on ", self.hparams.labels.num_train, " in batches of ", per_rank_batch_size)
        for start_index in range(0, self.hparams.labels.num_train, self.hparams.batch_size):
            print(f"debug. start_index: {start_index}, step: {utils.exact_div(start_index, self.hparams.batch_size)}")
            end_index = start_index + self.hparams.batch_size
            all_ranks_indices = train_indices[start_index:end_index]    # 所有rank总共batch_size条训练数据
            our_indices = all_ranks_indices[self.rank::self.num_ranks]  # 每个rank分到per_rank_batch_size条训练数据
            lr = (1 - start_index / self.hparams.labels.num_train) * self.hparams.lr # 学习率线性衰减
            self.train_batch(our_indices, lr)

        # 4. 训练后归一化normalize_after
        if self.hparams.normalize_after:
            target_mean, target_std = np.zeros([]), np.ones([])
            self.normalize(self.sample_policy_responses, target_mean, target_std)


def train(hparams: HParams):
    """
    hparams: 整个RLHF训练任务超参数, 包括ppo, rewards, run, task四大部分
    这是训练reward模型的入口
    初始化环境 → 构建模型组件 → 设置保存逻辑 → 初始化变量 → 同步参数 → 训练 → 保存checkpoint
    """
    with tf.Graph().as_default():
        hyperparams.dump(hparams)
        utils.set_mpi_seed(hparams.run.seed)

        m = trained_models.TrainedModel(hparams.task.policy.initial_model)
        encoder = m.encoding.get_encoder() # ReversibleEncoder类的实例
        
        # Model parameters
        hyperparams.dump(m.hparams(), name='model_hparams')

        # 管理多进程通信
        comm = MPI.COMM_WORLD
        
        # 主要用于生成候选和对候选计算各类指标(logprob、熵、价值等)
        ref_policy = Policy(
            m, 
            scope='ref_policy',
            is_root=comm.Get_rank() == 0,
            embed_queries=lm_tasks.query_formatter(hparams.task, encoder),
            temperature=hparams.task.policy.temperature,
            build_respond=False
        )

        # 1. 创建模型封装器（第一个类）
        reward_model = rewards.RewardModelWrapper(m, is_root=comm.Get_rank() == 0)
        
        # 负责生成训练用的prompt(queries)
        query_sampler = lm_tasks.make_query_sampler(
            hparams=hparams.task, 
            encoder=encoder, 
            comm=comm,
            batch_size=utils.exact_div(hparams.rollout_batch_size, comm.Get_size())
        )

        tf.train.create_global_step()

        # 2. 创建训练器(第二个类), 传入模型封装器
        reward_trainer = RewardModelTrainer(
            reward_model=reward_model,
            policy=ref_policy,
            query_sampler=query_sampler,
            hparams=hparams,
            comm=comm,
        )

        # reward模型保存路径
        save_dir = hparams.run.save_dir

        # 仅在chief worker保存模型检查点
        if comm.Get_rank() == 0 and save_dir:
            print(f"Will save to {save_dir}")
            # save_dir: /tmp/save/train_reward/testdesc-2601032319
            saver = tf.train.Saver(max_to_keep=20, save_relative_paths=True)
            checkpoint_dir = os.path.join(save_dir, 'reward_model/checkpoints/model.ckpt')
            print(f"checkpoint_dir: {checkpoint_dir}")
            # checkpoint_dir: /tmp/save/train_reward/testdesc-2601032319/reward_model/checkpoints/model.ckpt

            if not save_dir.startswith('gs://'):
                os.makedirs(os.path.join(save_dir, 'reward_model'), exist_ok=True)
            
            with tf.gfile.Open(os.path.join(save_dir, 'train_reward_hparams.json'), 'w') as f:
                json.dump(hparams.to_nested_dict(), f, indent=2)
            
            with tf.gfile.Open(os.path.join(save_dir, 'reward_model', 'hparams.json'), 'w') as f:
                json.dump(reward_model.hparams.to_nested_dict(), f, indent=2)
            
            with tf.gfile.Open(os.path.join(save_dir, 'reward_model', 'encoding'), 'w') as f:
                json.dump(reward_model.trained_model.encoding.name, f, indent=2) # "main"
        else:
            saver = None
            checkpoint_dir = None

        with utils.variables_on_gpu():
            # tf.group()将多个操作组合成单一操作, 执行时各个操作并行执行(这里用于同时更新两个归一化参数)
            init_ops = tf.group(
                tf.global_variables_initializer(),      # 全局变量(权重和偏置)
                tf.local_variables_initializer(),       # 局部变量(训练循环中用于内部计算的计数器等) 
                summary.summary_writer_initializer_op() # 初始化TensorBoard摘要写入器(各种日志, 包括Loss曲线等)
            )

            @utils.graph_function()
            def sync_models():
                """
                在多进程训练中, 将rank0的参数广播到其他进程
                这里同步ref_policy+reward_model的所有参数
                """
                return utils.variable_synchronizer(
                    comm, 
                    vars=ref_policy.get_params() + reward_model.get_params()
                )

        # 冻结计算图，防止意外修改
        tf.get_default_graph().finalize()
        with utils.mpi_session() as sess:
            init_ops.run()          # 初始化变量
            sync_models()           # 同步参数到所有进程
            reward_trainer.train()  # 核心训练循环
            if saver:               # 保存最终模型
                saver.save(sess, checkpoint_dir)
