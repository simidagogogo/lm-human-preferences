"""
Synthetic scores.
合成得分
""" 

import os

import tensorflow as tf
from mpi4py import MPI

from lm_human_preferences.language import trained_models, model
from lm_human_preferences.utils import core as utils
from lm_human_preferences.utils.core import Schema


class RewardModelWrapper:
    """
    combine this with TrainedRewardModel
    本质：这是一个 Reward Model 的封装器/推理器
    """
    def __init__(
        self,
        trained_model, 
        *,
        scope='reward_model', 
        use_resource=False,
        is_root=True,
    ):
        # 预训练模型(用于给reward模型迁移权重)
        self.trained_model = trained_model
        self.hparams = trained_model.hparams()
        self.is_root = is_root
        self.use_resource = use_resource

        # 可逆分词器与编码器(ReversibleEncoder类的实例)
        self.encoder = self.trained_model.encoding.get_encoder()
        self.scope = scope
        
        # reward模型(随训练过程调整权重)
        self.model = model.Model(
            hparams=self.hparams, 
            scope=f'{scope}/model', 
            scalar_heads=['reward']
        )

        self.built = False
        self.padding_token = self.encoder.padding_token

        self.get_rewards = utils.graph_function(
            queries=Schema(tf.int32, (None, None)),
            responses=Schema(tf.int32, (None, None)),
        )(self.get_rewards_op)


    def get_encoder(self):
        # 可逆分词器与编码器(ReversibleEncoder类的实例)
        return self.encoder


    def _build(self, tokens, do_dropout=False, name=None):
        """
        奖励模型前向传播, 预测奖励值
        """
        with tf.variable_scope(
            self.scope, 
            reuse=self.built, 
            auxiliary_name_scope=not self.built, 
            use_resource=self.use_resource
        ):
            # 模型推理
            lm_output = self.model(
                X=tokens, 
                do_dropout=do_dropout,
                padding_token=self.padding_token
            )

            # 先取reward字段, 然后取序列最后一个时间步[:, -1], 即每条输入数据的最后一步的奖励
            reward = lm_output['reward'][:, -1]
            with tf.variable_scope('reward_norm'):
                # 如果还没有build则给初始值
                if not self.built:
                    self.reward_gain = tf.get_variable('gain', shape=(), initializer=tf.constant_initializer(1))
                    self.reward_bias = tf.get_variable('bias', shape=(), initializer=tf.constant_initializer(0))
                    self._reward_gain_p = tf.placeholder(name='gain_p', dtype=tf.float32, shape=())
                    self._reward_bias_p = tf.placeholder(name='bias_p', dtype=tf.float32, shape=())
                    self._set_reward_norm = tf.group(
                        self.reward_gain.assign(self._reward_gain_p),
                        self.reward_bias.assign(self._reward_bias_p)
                    )
                if reward is not None:
                    reward = self.reward_gain * reward + self.reward_bias
            
            if not self.built:
                self._set_initializers()
            
            # 推理后build置为True
            self.built = True
            return reward

    def ensure_built(self):
        """
        如果已经build则直接返回, 否则运行一次模型推理后会将build改为True
        """
        if self.built:
            return

        with tf.name_scope('dummy'):
            self._build(tokens=tf.zeros([0,0], dtype=tf.int32))

    def get_params(self):
        """
        获取权重前, 必须确保已经build
        """
        self.ensure_built()
        return self.model.get_params() + [self.reward_gain, self.reward_bias]

    def reset_reward_scale(self):
        sess = tf.get_default_session()
        sess.run(
            self._set_reward_norm, 
            feed_dict={self._reward_gain_p: 1, self._reward_bias_p: 0}
        )

    def set_reward_norm(self, *, old_mean, old_std, new_mean, new_std):
        """
        Given old_mean+-old_std of reward_model, change gain and bias to get N(new_mean,new_std).
        """
        sess = tf.get_default_session()
        old_gain, old_bias = sess.run((self.reward_gain, self.reward_bias))
        assert old_gain == 1 and old_bias == 0,\
            f'set_reward_norm expects gain = 1 and bias = 0, not {old_gain}, {old_bias}'
        # gain * N(old_mean,old_std) + bias = N(gain * old_mean, gain * old_std) + bias
        #                                   = N(gain * old_mean + bias, gain * old_std)
        # gain * old_std = new_std, gain = new_std / old_std
        # gain * old_mean + bias = new_mean, bias = new_mean - gain * old_mean
        gain = new_std / old_std
        bias = new_mean - gain * old_mean
        sess.run(self._set_reward_norm, feed_dict={self._reward_gain_p: gain, self._reward_bias_p: bias})

    def _set_initializers(self):
        """
        Change initializers to load a language model from a tensorflow checkpoint.
        Note: 不是从已训练好的rewoard模型ckpt从迁移, 而是从预训练模型的chpt中迁移
        # Skip if
        # 1. We're not rank 0.  Values will be copied from there.
        # 2. We want random initialization.  Normal initialization will do the work.
        """
        if not self.is_root:
            return
        
        if self.trained_model.name == 'test':
            return

        # 用于确保以下操作发生在变量初始化阶段, 避免与Graph构建的其它操作冲突
        with tf.init_scope():
            # Initialize!
            params = {v.op.name: v for v in utils.find_trainable_variables(self.scope)}
            
            # 如果变量字典为空则报错, 防止遗漏
            assert params
            
            # 用训练好的模型(self.trained_model)的init_op操作, 把checkpoint权重数据加载到当前变量params里, 变量命名空间为scope
            # 这样当前模型或policy对象参数不再是随机值，而是拥有主模型的学习参数。
            self.trained_model.init_op(params, new_scope=self.scope)

    def get_rewards_op(self, queries, responses):
        """
        给定一组prompt-token(queries)和一组response-token(responses)合成完整输入后, 送入rewardModel得到奖励分数
        """
        tokens = tf.concat([queries, responses], axis=1)
        return self._build(tokens)


class TrainedRewardModel():
    """
    训练好的奖励模型, 从检查点加载权重. 仅用于train_policy中
    哪里修改build的状态?
    """
    def __init__(
        self, 
        train_dir, 
        encoding, 
        *, 
        scope='reward_model', 
        comm=MPI.COMM_WORLD
    ):
        self.train_dir = train_dir
        self.comm = comm
        self.encoding = encoding
        encoder = encoding.get_encoder()
        
        if train_dir != 'test':
            self.hparams = trained_models.load_hparams(os.path.join(train_dir, 'hparams.json'))
            assert self.hparams.n_vocab == encoding.n_vocab, f'{self.hparams.n_vocab} != {encoding.n_vocab}'
        else:
            self.hparams = trained_models.test_hparams()

        self.padding_token = encoder.padding_token
        self.encoder = encoder
        self.scope = scope
        
        # reward模型
        self.model = model.Model(
            hparams=self.hparams, 
            scope=f'{scope}/model', 
            scalar_heads=['reward']
        )


    def _build(self, X):
        """
        奖励模型前向传播
        :param X: 输入数据
        :return: reward score
        """
        results = self.model(X=X, padding_token=self.padding_token)
        reward = results['reward'][:, -1] # 
        
        # 对奖励进行归一化
        with tf.variable_scope(f'{self.scope}/reward_norm'):
            self.reward_gain = tf.get_variable('gain', shape=(), initializer=tf.constant_initializer(1))
            self.reward_bias = tf.get_variable('bias', shape=(), initializer=tf.constant_initializer(0))
        reward = self.reward_gain * reward + self.reward_bias
        self._set_initializers()
        return reward


    def ensure_built(self):
        """
        如果已经build则直接返回, 否则运行一次模型推理后会将build改为True
        """
        if self.model.built:
            return

        with tf.name_scope('dummy'):
            self._build(X=tf.zeros([0,0], dtype=tf.int32))


    def _set_initializers(self):
        """
        Change initializers to load a model from a tensorflow checkpoint.
        Note: 必须完成build
        """
        if self.comm.Get_rank() > 0 or self.train_dir == 'test':
            return

        assert self.model.built

        checkpoint_scope = 'reward_model'
        with tf.init_scope():
            # Initialize!
            params = {v.op.name: v for v in self.get_params()}
            checkpoint = tf.train.latest_checkpoint(os.path.join(self.train_dir, 'checkpoints/'))
            available = tf.train.list_variables(checkpoint)
            unchanged = {}

            for name, shape in available:
                if not name.startswith(checkpoint_scope + '/'):
                    # print('skipping', name)
                    continue
                if name.endswith('adam') or name.endswith('adam_1'):
                    # print('skipping', name)
                    continue
                print('setting', name)
                var = params[self.scope + name[len(checkpoint_scope):]]
                assert var.shape == shape, 'Shape mismatch: %s.shape = %s != %s' % (var.op.name, var.shape, shape)
                unchanged[name] = var
            tf.train.init_from_checkpoint(checkpoint, unchanged)


    def get_params(self):
        return self.model.get_params() + [self.reward_gain, self.reward_bias]


    def score_fn(self, queries, responses):
        """
        """
        tokens = tf.concat([queries, responses], axis=1)
        return self._build(tokens)
