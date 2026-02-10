import tensorflow as tf

from lm_human_preferences.language import model, sample
from lm_human_preferences.utils import core as utils
from lm_human_preferences.utils.core import Schema


class Policy:
    """
    封装了Policy模型的高层接口, 专用于生成任务和RLHF
    
    主要职责
    1)模型参数与scope管理
    2)decode采样生成、评估分析reward/value等两大功能
    3)支持分布式、resource变量、训练模型参数初始化
    4)对外暴露便捷接口支持批量调用和TensorFlow原生操作
    
    应用场景
    1)自然语言生成/对话系统(结合人类偏好做强化训练等)
    2)RLHF流程中, 作为policy参与 采样响应 与 策略评估
    """
    def __init__(
        self,
        trained_model, 
        *,
        scope=None, 
        use_resource=False,
        embed_queries=lambda queries: queries,
        temperature=1.0, 
        is_root=True,
        build_respond=True
    ):
        """
        @trained_model: 已经训练好的模型对象, 封装了编码器、超参数、初始化等接口, 本身并不直接forward输入
        @scope: 变量作用域名,控制变量的命名空间及共享
        @use_resource: 是否用resource变量
        @embed_queries: 对输入queries进行变换(默认不变)
        @temperature: 采样温度,控制生成时的多样性(越大越随机)
        @is_root: 是否为主节点,和参数初始化有关(分布式场景)
        @build_respond: False表示该参考策略只用于评估而非实际生成
        """
        self.trained_model = trained_model
        self.model_hparams = trained_model.hparams()
        self.encoder = self.trained_model.encoding.get_encoder()
        self.is_root = is_root
        self.use_resource = use_resource

        with tf.variable_scope(scope, 'transformer_policy', use_resource=self.use_resource) as s:
            self.scope = s
            # 初始化时, 将 self.trained_model提供的预训练权重load到self.model(通过_set_initializers完成)
            # 在推理/采样/分析环节，都是用self.model. 输入tokens, 输出logits、value等
            self.model = model.Model(
                hparams=self.model_hparams,
                scalar_heads=['value']
            )

        self.built = False
        self.embed_queries = embed_queries
        self.temperature = temperature
        self.padding_token = self.encoder.padding_token

        if build_respond:
            self.respond = utils.graph_function(
                queries=Schema(tf.int32, (None, None)),
                length=Schema(tf.int32, ()),
            )(self.respond_op)
        
        self.analyze_responses = utils.graph_function(
            queries=Schema(tf.int32, (None, None)),
            responses=Schema(tf.int32, (None, None)),
        )(self.analyze_responses_op)


    def get_encoder(self):
        """
        返回模型编码器tokenizer, 用于字符串和token的转换
        """
        return self.encoder


    def step_core(
        self, 
        model_hparams, 
        tokens, 
        past=None, 
        past_tokens=None, 
        do_dropout=False, 
        name=None
    ):
        """
        模型前向推理. 支持处理历史上下文past_tokens, 生成logits、value、past(KV缓存)
        
        @hparams: 超参数
        @tokens: 当前步的输入tokens
        @past: 缓存了之前所有时间步计算出的Key和Value矩阵. 模型只需要计算当前最新输入token的Key和Value, 然后拼接到缓存中即可(常数级)
        @past_tokens: 所有历史输入tokens
        """
        with tf.name_scope(name, 'step'):
            # 手写TF静态图模型时, 实现权重共享的标准写法: 第一次调用创建新变量, 后续调用复用变量(共享权重)
            with tf.variable_scope(
                    self.scope,
                    reuse=self.built,
                    auxiliary_name_scope=not self.built,
                    use_resource=self.use_resource
                ):
                lm_output = self.model(
                    X=tokens, 
                    past=past, 
                    past_tokens=past_tokens,
                    do_dropout=do_dropout, 
                    padding_token=self.padding_token
                )

                # need to slice logits since we don't want to generate special tokens
                # logits: (batch, sequence, vocab_n)
                logits = lm_output['lm_logits'][:, :, :self.model_hparams.n_vocab]
                # kvcache. (batch, layers, 2, heads, sequence, features)
                presents = lm_output['present']
                # 价值预测. (batch, sequence)
                value = lm_output['value']
                
                if not self.built:
                    self._set_initializers()
                self.built = True
                
                print(f"[step_core]. logits.shape: {logits.shape}, presents.shape: {presents.shape}, value.shape: {value.shape}")
                return {
                    'logits': logits,
                    'values': value,
                    'presents': presents,
                }


    def ensure_built(self):
        """
        保证计算图(及变量)已被build
        若未build, 则以空输入“假build”一遍, 用于后续参数操作
        """
        if not self.built:
            with tf.name_scope('dummy'):
                self.step_core(self.model_hparams, tokens=tf.zeros([0, 0], dtype=tf.int32))


    def get_params(self):
        """
        获取策略网络下所有可训练参数变量. 先确保网络已build
        若变量为零说明未build, 直接断言异常
        """
        self.ensure_built()
        params = utils.find_trainable_variables(self.scope.name)
        assert len(params) > 0
        return params


    def _set_initializers(self):
        """
        Change initializers to load a language model from a tensorflow checkpoint.
        """
        # Skip if
        # 1. We're not rank 0.  Values will be copied from there.
        # 2. We want random initialization.  Normal initialization will do the work.
        
        # 主节点且非测试时加载权重
        # Note: 在分布式训练中只有主节点负责参数初始化, 其他副本等待同步, 避免多次重复初始化
        if not self.is_root or self.trained_model.name == 'test':
            return

        with tf.init_scope():
            scope = self.scope.name
            # Initialize!
            params = {v.op.name: v for v in utils.find_trainable_variables(scope)}
            self.trained_model.init_op(params, new_scope=scope)


    def respond_op(self, queries, length):
        """
        按输入的queries和采样长度, 生成模型的输出文本序列. 常用于生成式任务
        """
        contexts = self.embed_queries(queries)
        context_length = tf.shape(contexts)[1]
        print(f"[respond_op]. contexts.shape: {contexts.shape}, context_length: {context_length}")
        
        # 调用sample_sequence实现policy采样, 除了返回tokens(输出文本token), 还有概率对数logprobs和价值value
        result = sample.sample_sequence(
            step=self.step_core,
            context=contexts,
            length=length,
            model_hparams=self.model_hparams,
            temperature=self.temperature,
            extra_outputs={'values': tf.float32},
        )
        return dict(
            responses=result['tokens'][:, context_length:], # (batch_size, length). 仅取response部分
            logprobs=result['logprobs'],                    # (batch_size, length)
            values=result['values'],                        # (batch_size, length)
        )


    def analyze_responses_op(self, queries, responses):
        """
        基于一批 queries上下文 和 responses回复/候选 进行前向推断, 分析模型输出的各类指标(logprob、熵、价值等)以便于奖励建模或行为评估
        """
        contexts = self.embed_queries(queries)
        context_length = tf.shape(contexts)[1]
        tokens = tf.concat([contexts, responses], axis=1)
        result = self.step_core(self.model_hparams, tokens)
        logits = result['logits'][:, context_length-1:-1]

        logits /= self.temperature
        logprobs = utils.logprobs_from_logits(logits=logits, labels=responses)
        entropies = utils.entropy_from_logits(logits)
        
        values = result['values'][:, context_length-1:-1]
        print(f"[analyze_responses_op]. logits.shape: {logits.shape}, values.shape: {values.shape}")
        print(f"[analyze_responses_op]. logprobs: {logprobs}, entropies: {entropies}")
        
        return dict(
            logprobs=logprobs,
            entropies=entropies,
            values=values,
        )
