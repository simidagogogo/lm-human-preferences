#!/usr/bin/env python3

"""
(base) root@iZ0jlfyn5du7ptefx2tr5vZ:~/PycharmProjects/lm-human-preferences/lm_human_preferences# pipenv run ./test_train_reward.py 
name: main, self.base_path: /root/gpt-2-models/encodings/main
name: main, self.encoder_path: /root/gpt-2-models/encodings/main/encoder.json
name: main, self.bpe_path: /root/gpt-2-models/encodings/main/vocab.bpe
name: test, self.base_path: gs://gpt-2/encodings/test
"""

import tempfile
from lm_human_preferences import train_reward

def hparams_for_test():
    hparams = train_reward.HParams()
    hparams.rollout_batch_size = 8
    hparams.task.query_length = 2
    hparams.task.response_length = 3
    hparams.noptepochs = 1
    hparams.task.policy.initial_model = 'test'
    hparams.task.query_dataset = 'test'
    hparams.task.start_text = None
    hparams.run.log_interval = 1

    hparams.labels.source = 'test'
    hparams.labels.num_train = 16
    hparams.labels.type = 'best_of_4'

    hparams.batch_size = 8

    return hparams


def train_reward_test(override_params):
    hparams = hparams_for_test()
    hparams.override_from_dict(override_params)
    hparams.validate()
    train_reward.train(hparams=hparams)


def test_basic():
    train_reward_test({})


def test_scalar_compare():
    train_reward_test({'labels.type': 'scalar_compare'})


def test_scalar_rating():
    train_reward_test({'labels.type': 'scalar_rating'})


def test_normalize_before():
    train_reward_test({
        'normalize_before': True,
        'normalize_after': False,
        'normalize_samples': 1024,
        'debug_normalize': 1024,
    })


def test_normalize_both():
    train_reward_test({
        'normalize_before': True,
        'normalize_after': True,
        'normalize_samples': 1024,
        'debug_normalize': 1024,
    })

def test_save():
    train_reward_test({
        'run.save_dir': tempfile.mkdtemp() ,
        'run.save_interval': 1
    })
