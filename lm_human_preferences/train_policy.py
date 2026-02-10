#!/usr/bin/env python3

"""
(base) root@iZ0jlfyn5du7ptefx2tr5vZ:~/PycharmProjects/lm-human-preferences/lm_human_preferences# pipenv run ./test_train_policy.py 
name: main, self.base_path: /root/gpt-2-models/encodings/main
name: main, self.encoder_path: /root/gpt-2-models/encodings/main/encoder.json
name: main, self.bpe_path: /root/gpt-2-models/encodings/main/vocab.bpe
name: test, self.base_path: gs://gpt-2/encodings/test
"""

import json
import os
import sys
import time
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

from lm_human_preferences import lm_tasks, train_reward
from lm_human_preferences.language import trained_models
from lm_human_preferences.policy import Policy
from lm_human_preferences.rewards import TrainedRewardModel
from lm_human_preferences.utils import core as utils
from lm_human_preferences.utils import hyperparams
from lm_human_preferences.utils.core import Schema


@dataclass
class AdaptiveKLParams(hyperparams.HParams):
    """
    在强化学习策略优化PPO中, KL target可以约束新旧策略的偏离程度(防止策略更新过快或不稳定), horizon决定调整的周期
    如果实际KL偏离目标太多, 则可以动态调整更新步长，保证训练稳定
    
    KL散度常用于衡量两个概率分布之间的“距离”.
    在强化学习、变分自编码器或强化学习RLHF相关算法中, 经常要调整KL散度, 使其维持在某个目标水平, 以保持生成策略的多样性和与参考策略的距离合适
    """
    target: float = None
    horizon: int = 10000  # in episodes


@dataclass
class RewardHParams(hyperparams.HParams):
    """
    奖励函数相关参数
    """
    kl_coef: float = 0.2                                    # KL散度系数，正则化奖励以避免策略与参考策略偏离过大
    adaptive_kl: Optional[AdaptiveKLParams] = None          # 自适应KL参数，可选；控制动态调整KL系数
    trained_model: Optional[str] = None                     # 训练模型的路径/标识(用于直接加载)
    train_new_model: Optional[train_reward.HParams] = None  # 新模型的训练参数（若需重新训练）

    def validate(self, *, prefix=''):
        """
        必须在“已有模型(trained_model)”与“用新参数训练新模型(train_new_model)”两者中选择其一，
        不能同时选择，也不能两者都不选，否则训练/推理过程无法使用奖励模型。
        """
        super().validate(prefix=prefix)
        assert self.trained_model is None or self.train_new_model is None, 'Cannot use trained_model and train new model'
        assert self.trained_model is not None or self.train_new_model is not None, 'Need either trained_model or to train a new model'

    
@dataclass
class PpoHParams(hyperparams.HParams):
    total_episodes: int = 2000000  # 总训练回合数（采集的数据总量），决定整体训练多充分
    batch_size: int = 64           # 每个更新周期（epoch）的采样批次大小
    nminibatches: int = 1          # 一个 batch 被分成几个mini batch进行多次梯度更新
    noptepochs: int = 4            # 每个 batch 被重复优化几次（epoch），即在同一数据上用策略优化器走多少遍
    lr: float = 5e-6               # 学习率，影响每一步参数调整的幅度
    vf_coef: float = .1            # Value函数（价值网络）损失的系数，在损失函数中占多大比重
    cliprange: float = .2          # 策略更新时的裁剪范围，防止新旧策略差异过大，保障训练稳定
    cliprange_value: float = .2    # 价值网络更新的裁剪范围，防止value估值剧烈震荡
    gamma: float = 1               # 奖励的折扣因子，值越小越关注短期回报，越大越关注长远回报
    lam: float = 0.95              # 广义优势估计（GAE）的lambda值，权衡短期与长期优势
    whiten_rewards: bool = True    # 是否对奖励进行标准化（归一化到均值为0、方差为1），有助于训练稳定


@dataclass
class HParams(hyperparams.HParams):
    """
    PPO训练整体的高层参数设置
    """
    run: train_reward.RunHParams = field(default_factory=train_reward.RunHParams)  # 训练运行相关参数
    task: lm_tasks.TaskHParams = field(default_factory=lm_tasks.TaskHParams)       # 语言模型任务相关参数
    rewards: RewardHParams = field(default_factory=RewardHParams)                  # 奖励函数相关参数
    ppo: PpoHParams = field(default_factory=PpoHParams)                            # PPO算法相关参数

    def validate(self, *, prefix=''):
        # 调用父类的超参数校验
        super().validate(prefix=prefix)
        
        # 计算minibatch大小: 必须能被nminibatches整除
        # NOTE: 这里需要除以总ranks数（分布式训练用，暂未体现）
        minibatch_size = utils.exact_div(self.ppo.batch_size, self.ppo.nminibatches)
        if self.ppo.whiten_rewards:
            # PPO的whiten_rewards依赖较大样本，若最小批次太小会报错
            assert minibatch_size >= 8, f"Minibatch size {minibatch_size} is insufficient for whitening in PPOTrainer.loss"


def nupdates(hparams):
    return utils.ceil_div(hparams.ppo.total_episodes, hparams.ppo.batch_size)


def policy_frac(hparams):
    """How far we are through policy training."""
    return tf.cast(tf.train.get_global_step(), tf.float32) / nupdates(hparams)


def tf_times():
    """Returns (time since start, time since last) as a tensorflow op."""
    # Keep track of start and last times
    with tf.init_scope():
        init = tf.timestamp()

    def make(name):
        return tf.Variable(init, name=name, trainable=False, use_resource=True)

    start = make('start_time')
    last = make('last_time')

    # Get new time and update last
    now = tf.timestamp()
    prev = last.read_value()
    with tf.control_dependencies([prev]):
        with tf.control_dependencies([last.assign(now)]):
            return tf.cast(now - start.read_value(), tf.float32), tf.cast(now - prev, tf.float32)


class FixedKLController:
    """
    固定KL系数控制器
    固定KL惩罚项的权重系数, 从初始化到类的整个生命周期不变化, 不随训练动态调整
    """
    def __init__(self, kl_coef):
        # 初始KLloss权重系数
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


class AdaptiveKLController:
    """
    自适应KL系数控制器
    每隔一段时间调用update()以动态调整KL惩罚项的权重系数, 使训练时KL散度能够自动保持在一个合理的目标范围内, 从而兼顾模型的探索与稳定性
    """
    def __init__(self, init_kl_coef, hparams):
        self.value = init_kl_coef
        self.hparams = hparams

    def update(self, current, n_steps):
        """
        update方法在训练过程中定期被调用, 用于根据最近的KL散度动态调整惩罚权重
        @current: 当前迭代步骤实际测量到的KL散度
        @n_steps: 距离上次更新已过去多少训练步骤
        """
        target = self.hparams.target
        
        # 计算当前KL与目标值之间的相对误差
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        
        """
        这个式子决定KL系数该按怎样的比例调整。
        如果当前KL > 目标KL, 则 proportional_error > 0, mult > 1, KL系数增大, 惩罚加重, 模型被拉回去
        如果当前KL < 目标KL, 则 proportional_error < 0, mult < 1, KL系数减小, 惩罚变轻, 模型能自由探索
        n_steps / horizon 控制调整的速度(步数多, 调整大)
        """
        mult = 1 + proportional_error * n_steps / self.hparams.horizon
        self.value *= mult


class PPOTrainer():
    """
    基于TF实现的PPO(Proximal Policy Optimization)训练器, 专门用于RLHF场景, 即利用强化学习微调语言模型
    核心逻辑
        Policy(当前模型)根据Query(提示词)生成Response(回复), 
        然后通过Score Function(奖励模型)打分, 并结合与 Ref Policy(参考模型)的KL散度作为约束, 计算最终奖励
        最后使用PPO算法更新模型参数
        
    这个类是一个标准的、工业级的PPO-RLHF训练器实现.
    它处理了从数据采样、奖励计算、GAE广义优势估计到梯度更新的完整流程, 并包含了PPO算法特有的Clipping机制 和 KL惩罚机制, 
    以确保语言模型在微调过程中既能满足人类偏好(高Score), 又不会崩溃或遗忘原有的语言能力(低KL)
    """
    def __init__(self, *, policy, ref_policy, query_sampler, score_fn, hparams, comm):
        """
        负责设置训练所需的组件和超参数
        @policy:        当前训练的模型
        @ref_policy:    参考模型，通常是SFT后的模型，参数冻结
        @query_sampler:
        @score_fn:      奖励函数
        @hparams:       超参数
        @comm:
        """
        self.comm = comm
        self.policy = policy
        self.ref_policy = ref_policy
        self.score_fn = score_fn
        self.hparams = hparams

        # KL控制器kl_ctl. 控制模型不要偏离参考模型太远
        if hparams.rewards.adaptive_kl is None:
            self.kl_ctl = FixedKLController(hparams.rewards.kl_coef)
        else:
            self.kl_ctl = AdaptiveKLController(
                hparams.rewards.kl_coef, 
                hparams=hparams.rewards.adaptive_kl
            )
        response_length = hparams.task.response_length
        query_length = hparams.task.query_length

        @utils.graph_function()
        def sample_queries():
            return query_sampler()['tokens']
        self.sample_queries = sample_queries

        def compute_rewards(scores, logprobs, ref_logprobs):
            """
            计算每一步的最终奖励和KL惩罚项.
            在RLHF经常会用奖励模型得分Scores和KL惩罚共同构成最终奖励, 引导生成模型“既符合人类偏好, 又不偏离原分布太远”
            
            @scores: 奖励模型得分（通常 shape是 [batch_size, 1] 或 [batch_size]，代表每个样本给出的 scalar 评分）
            @logprobs: 主模型生成每一步token的对数概率（可以是 [batch_size, seq_len] 的二维数组）
            @ref_logprobs: 参考模型（通常是未微调的原始模型）的每一步token对数概率（shape同上）
            
            
            该函数复合了惩罚生成模型与参考模型分布差异的KL损失，并在句末加上奖励模型得分，形成序列级的 RLHF 奖励结构。
            奖励分布体现出：鼓励模型别脱离“祖宗”（参考lm），但也要得到高人类指令得分。
            KL系数(self.kl_ctl.value)可根据训练状态动态调节（见之前 AdaptiveKLController）。
            
            可视化流程
            1.每步奖励 = -KL惩罚 （全部步骤都有）
            2.句末加奖励 = 上述 + scores
            3.最终奖励：除句末外各步只有惩罚，句末有奖励+惩罚。
            """
            
            """
            计算KL散度项：
            理论上我们用 KL(P‖Q)=E_P[log P(x) - log Q(x)]
            其中，P(x)：当前模型生成的概率，Q(x)：参考模型生成的概率
            这里直接用每一步 log 概率的差做近似（与序数、长度无关，通常会逐步累加），结果 shape 为 [batch_size, seq_len]
            本质上，这就是对每个 token，“主模型偏离参考模型的幅度”；数值越大，偏离越大。            
            """
            kl = logprobs - ref_logprobs
            
            """
            计算非评分奖励（KL惩罚项）：
            self.kl_ctl.value 是 KL 惩罚项的当前系数（通常会自适应调整）
            -self.kl_ctl.value * kl 意味着：
            如果主模型偏离参考模型（kl为正），就有负奖励（惩罚）
            如果主模型更靠近参考（kl为负），则奖励或惩罚减少
            non_score_reward 形状为 [batch_size, seq_len]
            """
            non_score_reward = -self.kl_ctl.value * kl
            
            """
            将奖励分配到最后一个 token（或序列末尾）：
            1.rewards = non_score_reward.copy()
                先准备一份“全部是KL惩罚项的奖励”
            2.rewards[:, -1] += scores
                对每个样本的“最后一个 token”（即 [:, -1]），额外加上奖励模型得分（scores）
                这样做的目的是：只有最后一个位置/整句才获得奖励模型的反馈，其余token仅有KL惩罚值
                常见于序列奖励场景，避免提前泄漏奖励
            """
            rewards = non_score_reward.copy()
            rewards[:, -1] += scores
            
            """
            rewards：最终的奖励（shape：[batch_size, seq_len]）
                包含KL惩罚项和在句末位置加上的奖励分数
            non_score_reward：纯粹的KL惩罚项
            self.kl_ctl.value：当前KL系数值，方便后续追踪自适应效果
            """
            return rewards, non_score_reward, self.kl_ctl.value
        self.compute_rewards = compute_rewards

        # per rank sizes
        per_rank_rollout_batch_size = utils.exact_div(hparams.ppo.batch_size, comm.Get_size())
        per_rank_minibatch_size = utils.exact_div(per_rank_rollout_batch_size, hparams.ppo.nminibatches)

        @utils.graph_function(rollouts=dict(
            queries=Schema(tf.int32, (per_rank_minibatch_size, query_length)),
            responses=Schema(tf.int32, (per_rank_minibatch_size, response_length)),
            values=Schema(tf.float32, (per_rank_minibatch_size, response_length)),
            logprobs=Schema(tf.float32, (per_rank_minibatch_size, response_length)),
            rewards=Schema(tf.float32, (per_rank_minibatch_size, response_length)),
        ))


        def train_minibatch(rollouts):
            """
            One step of PPO training.
            """
            left = 1 - policy_frac(hparams)
            lrnow = hparams.ppo.lr * left

            ppo_loss, stats = self.loss(rollouts)
            ppo_train_op = utils.minimize(
                loss=ppo_loss, 
                lr=lrnow, 
                params=policy.get_params(), 
                name='ppo_opt', 
                comm=self.comm
            )
            return ppo_train_op, stats


        def train(rollouts):
            """
            执行PPO的多次Epoch(noptepochs)
            将收集到的数据（Rollouts）打乱（Shuffle），分批次（Minibatch）喂给 train_minibatch 进行更新
            
            拿着这些数据（Query, Response, Reward, Values, Logprobs）去更新模型参数。
            """
            stat_list = []

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(hparams.ppo.noptepochs):
                order = np.random.permutation(per_rank_rollout_batch_size)
                for mb_start in range(0, per_rank_rollout_batch_size, per_rank_minibatch_size):
                    mb_data = {k: v[order[mb_start:mb_start+per_rank_minibatch_size]]
                               for k, v in rollouts.items()}

                    step = tf.train.get_global_step().eval()

                    _, stats = train_minibatch(mb_data)
                    stat_list.append(stats)

            # Collect the stats. (They will be averaged later.)
            return {k: [s[k] for s in stat_list] for k in stat_list[0].keys()}
        self.train = train

        # NOTE: must line up with stats created in self.loss (TODO: better solution?)
        scalar_batch = Schema(tf.float32, (None,))
        ppo_stat_schemas = utils.flatten_dict(dict(
            loss=dict(policy=scalar_batch, value=scalar_batch, total=scalar_batch),
            policy=dict(entropy=scalar_batch, approxkl=scalar_batch, clipfrac=scalar_batch),
            returns=dict(mean=scalar_batch, var=scalar_batch),
            val=dict(vpred=scalar_batch, error=scalar_batch, clipfrac=scalar_batch, mean=scalar_batch, var=scalar_batch),
        ), sep='/')
        
        stat_data_schemas = dict(
            logprobs=Schema(tf.float32, (None, hparams.task.response_length)),
            ref_logprobs=Schema(tf.float32, (None, hparams.task.response_length)),
            scores=scalar_batch,
            non_score_reward=Schema(tf.float32, (None, hparams.task.response_length)),
            score_stats=score_fn.stat_schemas,
            train_stats=ppo_stat_schemas,
        )
        
        @utils.graph_function(**stat_data_schemas, kl_coef=Schema(tf.float32, ()))
        def record_step_stats(*, kl_coef, **data):
            """
            计算并记录各种统计指标供 TensorBoard 查看：KL 均值、熵（Entropy）、奖励均值、Explained Variance（价值函数拟合程度）等。
            """
            ppo_summary_writer = utils.get_summary_writer(self.hparams.run.save_dir, subdir='ppo', comm=self.comm)

            kl = data['logprobs'] - data['ref_logprobs']
            mean_kl = tf.reduce_mean(tf.reduce_sum(kl, axis=1))
            mean_entropy = tf.reduce_mean(tf.reduce_sum(-data['logprobs'], axis=1))
            mean_non_score_reward = tf.reduce_mean(tf.reduce_sum(data['non_score_reward'], axis=1))
            stats = {
                'objective/kl': mean_kl,
                'objective/kl_coef': kl_coef,
                'objective/entropy': mean_entropy,
            }
            for k, v in data['train_stats'].items():
                stats[f'ppo/{k}'] = tf.reduce_mean(v, axis=0)
            for k, v in data['score_stats'].items():
                mean = tf.reduce_mean(v, axis=0)
                stats[f'objective/{k}'] = mean
                stats[f'objective/{k}_total'] = mean + mean_non_score_reward

            stats = utils.FlatStats.from_dict(stats).map_flat(
                partial(utils.mpi_allreduce_mean, comm=self.comm)).as_dict()

            # Add more statistics
            step = tf.train.get_global_step().read_value()
            stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
            steps = step + 1
            stats.update({
                'elapsed/updates': steps,
                'elapsed/steps/serial': steps * hparams.task.response_length,
                'elapsed/steps/total': steps * hparams.ppo.batch_size * hparams.task.response_length,
                'elapsed/episodes': steps * hparams.ppo.batch_size,
            })

            # Time statistics
            total, delta = tf_times()
            stats.update({
                'elapsed/fps': tf.cast(hparams.ppo.batch_size * hparams.task.response_length / delta, tf.int32),
                'elapsed/time': total,
            })
            if ppo_summary_writer:
                record_op = utils.record_stats(
                    stats=stats, summary_writer=ppo_summary_writer, step=step, log_interval=hparams.run.log_interval, name='ppo_stats', comm=self.comm)
            else:
                record_op = tf.no_op()
            return record_op, stats
        self.record_step_stats = record_step_stats


    def print_samples(self, queries, responses, scores, logprobs, ref_logprobs):
        """
        用于在控制台打印生成的文本样本，方便人工检查模型有没有胡言乱语，以及 KL 散度和分数的变化。
        """
        if self.comm.Get_rank() != 0:
            return
        if tf.train.get_global_step().eval() % self.hparams.run.log_interval != 0:
            return

        encoder = self.policy.encoder

        # Log samples
        for i in range(min(3, len(queries))):
            sample_kl = np.sum(logprobs[i] - ref_logprobs[i])
            print(encoder.decode(queries[i][:self.hparams.task.query_length]).replace("\n", "⏎"))
            print(encoder.decode(responses[i]).replace("\n", "⏎"))
            print(f"  score = {scores[i]:+.2f}")
            print(f"  kl = {sample_kl:+.2f}")
            print(f"  total = {scores[i] - self.hparams.rewards.kl_coef * sample_kl:+.2f}")


    def step(self):
        """
        这是外部调用的主函数，每调用一次 step() 就完成一轮完整的 "采样 -> 评估 -> 训练"。
        """
        step_started_at = time.time()
        
        # 1. 采样 Rollout
        # 获取一批Prompt
        queries = self.sample_queries()
        print(f"step_started_at: {step_started_at}, len(queries): {len(queries)}, queries: {queries}")
        
        # 模型根据Prompt生成回复, 得到logprobs
        rollouts = self.policy.respond(queries, length=self.hparams.task.response_length)
        responses = rollouts['responses']
        logprobs = rollouts['logprobs']
        rollouts['queries'] = queries
        
        # 2. 评估 (Reference & Score)
        # 让参考模型评估同一批回复，得到 ref_logprobs (参考概率)，用于计算 KL。
        ref_logprobs = self.ref_policy.analyze_responses(queries, responses)['logprobs']
        # 计算生成文本的任务分数Score
        scores, postprocessed_responses, score_stats = self.score_fn(queries, responses)

        # 3. 奖励计算
        # 结合 Score 和 KL 散度计算最终 Reward
        rewards, non_score_reward, kl_coef = self.compute_rewards(
            scores=scores,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs
        )
        rollouts['rewards'] = rewards

        # 4.训练 (Update)
        train_stats = self.train(rollouts=rollouts)

        # 5.后处理
        # 记录日志
        _, stats = self.record_step_stats(
            scores=scores, 
            logprobs=logprobs, 
            ref_logprobs=ref_logprobs, 
            non_score_reward=non_score_reward,
            train_stats=train_stats, 
            score_stats=score_stats, 
            kl_coef=kl_coef
        )

        # 更新 KL 系数
        self.kl_ctl.update(stats['objective/kl'], self.hparams.ppo.batch_size)

        # 打印样例
        self.print_samples(
            queries=queries, 
            responses=postprocessed_responses,
            scores=scores, 
            logprobs=logprobs, 
            ref_logprobs=ref_logprobs
        )

        # Record profiles of the step times
        step = tf.get_default_session().run(tf.train.get_global_step())
        step_time = time.time() - step_started_at
        eps_per_second = float(self.hparams.ppo.batch_size) / step_time
        if self.comm.Get_rank() == 0:
            print(f"[ppo_step {step}] step_time={step_time:.2f}s, "
                  f"eps/s={eps_per_second:.2f}")


    def loss(self, rollouts):
        """
        PPO损失函数计算
        这是PPO算法的数学核心, 包含三个部分: Policy Loss, Value Loss, Entropy Bonus
        """
        values = rollouts['values']
        old_logprob = rollouts['logprobs']
        rewards = rollouts['rewards']
        
        with tf.name_scope('ppo_loss'):
            if self.hparams.ppo.whiten_rewards:
                rewards = utils.whiten(rewards, shift_mean=False)

            lastgaelam = 0
            advantages_reversed = []
            gen_length = self.hparams.task.response_length
            
            """
            1. GAE (Generalized Advantage Estimation) 计算:
            作用：计算优势函数 (Advantage)，衡量当前动作比平均情况好多少。使用了 TD-error (delta) 和折扣因子 gamma、lam。
            """
            for t in reversed(range(gen_length)):
                nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = rewards[:, t] + self.hparams.ppo.gamma * nextvalues - values[:, t]
                lastgaelam = delta + self.hparams.ppo.gamma * self.hparams.ppo.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = tf.stack(advantages_reversed[::-1], axis=1)
            returns = advantages + values

            advantages = utils.whiten(advantages)
            advantages = tf.stop_gradient(advantages)  # Shouldn't do anything, but better not to think about it

            outputs = self.policy.analyze_responses_op(rollouts['queries'], rollouts['responses'])

            vpred = outputs['values']
            vpredclipped = tf.clip_by_value(vpred, values - self.hparams.ppo.cliprange_value, values + self.hparams.ppo.cliprange_value)
            vf_losses1 = tf.square(vpred - returns)
            vf_losses2 = tf.square(vpredclipped - returns)
            
            
            """
            2. Value Loss (价值损失):
            目标：让 Critic (Value Function) 预测的价值更接近真实的 Returns。
            Clipping: 使用了 vpredclipped，防止 Value Function 更新过猛。
            """
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            vf_clipfrac = tf.reduce_mean(
                tf.cast(tf.greater(vf_losses2, vf_losses1), tf.float32)
            )

            logprob = outputs['logprobs']
            ratio = tf.exp(logprob - old_logprob)
            pg_losses = -advantages * ratio
            pg_losses2 = -advantages * tf.clip_by_value(ratio, 1.0 - self.hparams.ppo.cliprange, 1.0 + self.hparams.ppo.cliprange)
            
            """
            3. Policy Loss (策略损失 - PPO核心):
            计算新旧策略的比率 ratio = exp(new_logprob - old_logprob)。
            PPO Clipping: 核心公式 min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)。如果更新幅度超过 cliprange，则截断梯度，保证训练稳定性。
            """
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            pg_clipfrac = tf.reduce_mean(tf.cast(tf.greater(pg_losses2, pg_losses), tf.float32))

            """
            5. 总损失:
             (通常熵会作为正则项加进去，或者只用来监控，这里代码里似乎没有直接加到 loss 变量里，可能是通过 policy.get_params() 的优化器设置处理，或者此实现隐式包含)。修正：细看代码，entropy确实只在stats里返回了，可能此实现未将熵加入Loss，或者在 pg_loss 计算外部有特殊处理。但在标准PPO中通常是 loss - ent_coef * entropy。
            """
            loss = pg_loss + self.hparams.ppo.vf_coef * vf_loss

            # 4. Entropy (熵)
            # 作用：鼓励模型保持一定的随机性，防止过早收敛（Mode Collapse）。
            entropy = tf.reduce_mean(outputs['entropies'])
            approxkl = .5 * tf.reduce_mean(tf.square(logprob - old_logprob))

            return_mean, return_var = tf.nn.moments(returns, axes=list(range(returns.shape.ndims)))
            value_mean, value_var = tf.nn.moments(values, axes=list(range(values.shape.ndims)))

            stats = dict(
                loss=dict(policy=pg_loss, value=vf_loss, total=loss),
                policy=dict(entropy=entropy, approxkl=approxkl, clipfrac=pg_clipfrac),
                returns=dict(mean=return_mean, var=return_var),
                val=dict(vpred=tf.reduce_mean(vpred), 
                         error=tf.reduce_mean((vpred - returns) ** 2),
                         clipfrac=vf_clipfrac, 
                         mean=value_mean, 
                         var=value_var
                    )
            )
            return loss, utils.flatten_dict(stats, sep='/')


def make_score_fn(hparams, score_model):
    """
    构建计算分数Score的逻辑, 通常代表人类偏好
    """
    padding_token = score_model.padding_token
    postprocess_fn = lm_tasks.postprocess_fn_from_hparams(hparams, padding_token)
    
    #decorate requires a named function, postprocess_fn can be anonymous
    @utils.graph_function(responses=Schema(tf.int32, (None, None)))
    def postprocess(responses):
        """"模型生成的tokenID进行后处理(例如去掉 padding、截断等)"""
        return postprocess_fn(responses)

    filter_fn = lm_tasks.filter_fn_from_hparams(hparams)
    @utils.graph_function(responses=Schema(tf.int32, (None, None)), rewards=Schema(tf.float32, (None,)))
    def penalize(responses, rewards):
        """
        过滤器逻辑。
        如果生成结果不符合规则(比如太短、含有敏感词), 则强制给一个惩罚分(penalty_reward_value), 覆盖掉原来的分数
        """
        valid = filter_fn(responses)
        return tf.where(
            valid, 
            rewards, 
            hparams.penalty_reward_value * tf.ones_like(rewards)
        )

    @utils.graph_function(queries=Schema(tf.int32, (None, None)), responses=Schema(tf.int32, (None, None)))
    def unpenalized_score_fn(queries, responses):
        """调用外部的评分模型(Reward Model)给生成结果打分"""
        return score_model.score_fn(queries, responses)

    def score_fn(queries, responses):
        responses = postprocess(responses)
        score = penalize(responses, unpenalized_score_fn(queries, responses))
        return score, responses, dict(score=score)
    score_fn.stat_schemas = dict(score=Schema(tf.float32, (None,)))
    return score_fn


def train(hparams: HParams):
    save_dir = hparams.run.save_dir
    if hparams.rewards.train_new_model:
        assert hparams.task == hparams.rewards.train_new_model.task, f'{hparams.task} != {hparams.rewards.train_new_model.task}'
        hparams.rewards.train_new_model.run.save_dir = save_dir
        train_reward.train(hparams.rewards.train_new_model)
        if 'pytest' in sys.modules:
            hparams.rewards.trained_model = 'test'
        elif save_dir:
            hparams.rewards.trained_model = None if save_dir is None else os.path.join(save_dir, 'reward_model')

    comm = MPI.COMM_WORLD

    with tf.Graph().as_default():
        hyperparams.dump(hparams)

        m = trained_models.TrainedModel(hparams.task.policy.initial_model)
        encoder = m.encoding.get_encoder()
        hyperparams.dump(m.hparams(), name='model_hparams')

        if save_dir:
            if not save_dir.startswith('https:'):
                os.makedirs(os.path.join(save_dir, 'policy'), exist_ok=True)
            with tf.gfile.Open(os.path.join(save_dir, 'train_policy_hparams.json'), 'w') as f:
                json.dump(hparams.to_nested_dict(), f, indent=2)
            with tf.gfile.Open(os.path.join(save_dir, 'policy', 'hparams.json'), 'w') as f:
                json.dump(m.hparams().to_nested_dict(), f, indent=2)
            with tf.gfile.Open(os.path.join(save_dir, 'policy', 'encoding'), 'w') as f:
                json.dump(m.encoding.name, f, indent=2)

        utils.set_mpi_seed(hparams.run.seed)
        
        # 
        score_model = TrainedRewardModel(
            hparams.rewards.trained_model, 
            m.encoding, 
            comm=comm
        )

        # 基线策略
        ref_policy = Policy(
            m, 
            scope='ref_policy',
            is_root=comm.Get_rank() == 0,
            embed_queries=lm_tasks.query_formatter(hparams.task, encoder),
            temperature=hparams.task.policy.temperature,
            build_respond=False
        )

        # 主策略
        policy = Policy(
            m, 
            scope='policy',
            is_root=comm.Get_rank() == 0,
            embed_queries=lm_tasks.query_formatter(hparams.task, encoder),
            temperature=hparams.task.policy.temperature,
            build_respond=True
        )

        query_sampler = lm_tasks.make_query_sampler(
            hparams=hparams.task, 
            encoder=encoder, 
            comm=comm,
            batch_size=utils.exact_div(hparams.ppo.batch_size, comm.Get_size()),
        )

        per_rank_minibatch_size = utils.exact_div(hparams.ppo.batch_size, hparams.ppo.nminibatches * comm.Get_size())
        if hparams.ppo.whiten_rewards:
            assert per_rank_minibatch_size >= 8, f"Per-rank minibatch size {per_rank_minibatch_size} is insufficient for whitening"

        global_step = tf.train.get_or_create_global_step()
        increment_global_step = tf.group(global_step.assign_add(1))

        with utils.variables_on_gpu():
            ppo_trainer = PPOTrainer(
                policy=policy, 
                ref_policy=ref_policy, 
                query_sampler=query_sampler,
                score_fn=make_score_fn(hparams.task, score_model=score_model),
                hparams=hparams, 
                comm=comm
            )

        if comm.Get_rank() == 0 and save_dir:
            print(f"Will save to {save_dir}")
            saver = tf.train.Saver(max_to_keep=20, save_relative_paths=True)
            checkpoint_dir = os.path.join(save_dir, 'policy/checkpoints/model.ckpt')
        else:
            saver = None
            checkpoint_dir = None

        @utils.graph_function()
        def sync_models():
            score_model.ensure_built()
            return utils.variable_synchronizer(comm, vars=score_model.get_params() + ref_policy.get_params() + policy.get_params())

        init_ops = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
            summary.summary_writer_initializer_op()
        )

        with utils.mpi_session() as sess:
            init_ops.run()
            sync_models()
            tf.get_default_graph().finalize()
            try:
                while global_step.eval() < nupdates(hparams):
                    ppo_trainer.step()
                    increment_global_step.run()
                    if saver and global_step.eval() % hparams.run.save_interval == 0:
                        saver.save(sess, checkpoint_dir, global_step=global_step)
            finally:
                if saver:
                    saver.save(sess, checkpoint_dir, global_step=global_step)
