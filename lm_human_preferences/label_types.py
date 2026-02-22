"""
Interface and implementations of label types for a reward model.

定义奖励模型人类标注类型LabelType的接口, 以及3种具体实现: 
  1. PickBest
  2. ScalarRating
  3. ScalarComparison
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
import tensorflow as tf
from lm_human_preferences.utils.core import Schema, pearson_r


class LabelType(ABC):
    """
    ABC(Abstract Base Classes)
    ABC用于定义 接口, 规定子类必须实现哪些方法, 且抽象类本身不能直接被实例化
    
    abstractmethod
    ABC强制约束每个子类都实现abstractmethod
    如果某个子类没有实现所有抽象方法, 那么该子类也仍然是ABC, 不能实例化
    """
    @abstractmethod
    def label_schemas(self) -> Dict[str, Schema]:
        """
        Schema for the human annotations.
        """
    
    @abstractmethod
    def question_schemas(self, *, query_length, response_length) -> Dict[str, Schema]:
        """
        Schema for the questions associated with this LabelType.
        """

    @abstractmethod
    def target_scales(self, labels: Dict[str, tf.Tensor]) -> Optional[tf.Tensor]:
        """
        label/目标值需要按某种尺度缩放或标准化(例如标量评分可能要做归一化)
        Extracts scalars out of labels whose scale corresponds to the reward model's output.
        May be none if the labels have no such information.
        """

    @abstractmethod
    def loss(self, reward_model, labels: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        loss
        :param labels: the questions with their labels
        :returns: a dict of stats, including 'loss' for the actual loss
        """


class PickBest(LabelType):
    """
    Pick best response amongst N.
    """
    def __init__(self, num_responses):
        self.num_responses = num_responses

    def label_schemas(self) -> Dict[str, Schema]:
        return dict(best=Schema(tf.int32, ()))

    def question_schemas(self, *, query_length, response_length) -> Dict[str, Schema]:
        return dict(
            query=Schema(tf.int32, (query_length,)),
            **{
                f"sample{i}": Schema(tf.int32, (response_length,)) 
                for i in range(self.num_responses)
            }
        )

    def target_scales(self, labels):
        return None

    def loss(self, reward_model, labels) -> Dict[str, tf.Tensor]:
        # logits: 多分类logit向量, 每一行对应一个样本, 每一列对应一个候选回答得分. [batch_size, num_responses]
        logits = tf.stack(
            [reward_model(labels['query'], labels[f'sample{i}']) for i in range(self.num_responses)], 
            axis=1
        )
        
        # logits: [batch_size, num_responses]
        # labels: [batch_size]
        # 对batch求平均, 得到标量loss
        error = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels['best'], logits=logits)
        )
        return dict(loss=error, error=error)


class ScalarRating(LabelType):
    """
    Rate a single number with a scalar score.
    """
    def __init__(self):
        pass

    def label_schemas(self):
        return dict(score=Schema(tf.float32, ()))

    def question_schemas(self, *, query_length, response_length) -> Dict[str, Schema]:
        return dict(
            query=Schema(tf.int32, (query_length,)),
            sample=Schema(tf.int32, (response_length,)),
        )
        
    def target_scales(self, labels):
        return labels['score']

    def loss(self, reward_model, labels):
        predicted = reward_model(labels['query'], labels['sample'])
        labels = labels['score']
        error = tf.reduce_mean((labels - predicted) ** 2, axis=0)
        label_mean, label_var = tf.nn.moments(labels, axes=[0])
        corr = pearson_r(labels, predicted)
        return dict(
            loss=error, 
            error=error,
            label_mean=label_mean, 
            label_var=label_var, 
            corr=corr
        )


class ScalarComparison(LabelType):
    """
    Give a scalar indicating difference between two responses.
    """
    def label_schemas(self):
        return dict(difference=Schema(tf.float32, ()))

    def question_schemas(self, *, query_length, response_length) -> Dict[str, Schema]:
        return dict(
            query=Schema(tf.int32, (query_length,)),
            sample0=Schema(tf.int32, (response_length,)),
            sample1=Schema(tf.int32, (response_length,)),
        )
    
    def target_scales(self, labels):
        # Divide by two to get something with the same variance as the trained reward model output
        return labels['difference'] / 2

    def loss(self, reward_model, labels):
        outputs0 = reward_model(labels['query'], labels['sample0'])
        outputs1 = reward_model(labels['query'], labels['sample1'])

        # 回归：拟合分数差
        # 让目标差值的方差 和 训练出来的reward model输出的方差对齐
        differences = labels['difference']
        predicted_differences = outputs1 - outputs0
        error = tf.reduce_mean((differences - predicted_differences) ** 2, axis=0)
        return dict(
            loss=error, 
            error=error
        )


def get(label_type: str) -> LabelType:
    if label_type == 'scalar_rating':
        return ScalarRating()
    if label_type == 'scalar_compare':
        return ScalarComparison()
    if label_type.startswith('best_of_'):
        n = int(label_type[len('best_of_'):])
        return PickBest(n)
    raise ValueError(f"Unexpected label type {label_type}")
