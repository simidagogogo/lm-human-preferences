#!/usr/bin/env python3

# Try to import mpi4py, if it fails, use our mock
try:
    import mpi4py
except ImportError:
    import sys
    import os
    # Add current directory to path and import mock
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mpi_mock
    sys.modules['mpi4py'] = mpi_mock
    sys.modules['mpi4py.MPI'] = mpi_mock.MPI

from lm_human_preferences.utils import launch
from lm_human_preferences.utils.combos import bind, combos, each, label, options_shortdesc, bind_nested
from lm_human_preferences import train_policy, train_reward


# 书籍任务配置 用于任务1,2
books_task = combos(
    bind('query_length', 64),
    bind('query_dataset', 'books'),
    bind('response_length', 24),
    bind('start_text', '.'),        # Start the context at the beginning of a sentence
    bind('end_text', '.'),          # End the context at the end of a sentence.
    bind('truncate_token', 13),     # Encoding of '.' -- end completions at the end of a sentence.
    bind('truncate_after', 16),     # Make sure completions are at least 16 tokens long.
    bind('policy.temperature', 0.7),
    bind('policy.initial_model', '124M'),
)

# 文本摘要配置 用于任务3
summarize_cnndm_task = combos(
    bind('query_prefix', 'Article:\n\n'),
    bind('query_suffix', '\n\nTL;DR:'),
    bind('end_text', '\n'),
    bind('query_dataset', 'cnndm'),
    bind('query_length', 500),
    bind('response_length', 75),
    bind('start_text', None),
    bind('truncate_after', 55),
    bind('truncate_token', 198),  # '\n'
    bind('policy.temperature', 0.5),
    bind('policy.initial_model', '124M'),
)

# TL;DR配置 用于任务4
summarize_tldr_task = combos(
    bind('query_suffix', '\n\nTL;DR:'),
    bind('query_dataset', 'tldr'),
    bind('query_length', 500),
    bind('response_length', 75),
    bind('start_text', None),
    bind('truncate_after', 55),
    bind('truncate_token', 198),  # '\n'
    bind('policy.temperature', 0.7),
    bind('policy.initial_model', '124M'),
)

def get_train_reward_experiments():
    """
    reward模型训练配置
    """
    _shared = combos(
        bind('labels.type', 'best_of_4'),
        bind('normalize_after', True),
        bind('normalize_before', True),
        bind('normalize_samples', 256),
    )

    _books_task = combos(
        bind_nested('task', books_task),
        _shared,
        bind('batch_size', 32),             # 训练奖励模型时的批次大小（用于梯度更新）
        bind('lr', 5e-5),
        bind('rollout_batch_size', 512),    # 从策略模型采样响应时的批次大小（用于生成训练数据）
    )

    # 任务1. 积极情感-风格化续写
    sentiment = combos(
        _books_task,
        bind('labels.source', 'https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/sentiment/offline_5k.json'),
        bind('labels.num_train', 4_992),
        bind('run.seed', 1)
    )

    # 任务2. 物理描述-风格化续写
    descriptiveness = combos(
        _books_task,
        bind('labels.source', 'https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/descriptiveness/offline_5k.json'),
        bind('labels.num_train', 4_992),
        bind('run.seed', 1)
    )

    # 任务3. CNN/Daily Mail摘要任务
    cnndm = combos(
        bind_nested('task', summarize_cnndm_task),
        _shared,
        # bind('labels.source', 'https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/cnndm/offline_60k.json'),
        # bind('labels.num_train', 60_000),
        bind('labels.source', 'https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/cnndm/online_45k.json'),
        bind('labels.num_train', 46_000),
        bind('batch_size', 2 * 8),
        bind('lr', 2.5e-5),
        bind('rollout_batch_size', 128),
        bind('run.seed', 1)
    )

    # 任务4. TL;DR摘要任务
    tldr = combos(
        bind_nested('task', summarize_tldr_task),
        _shared,
        # bind('labels.source', 'https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/tldr/offline_60k.json'),
        # bind('labels.num_train', 60_000),
        bind('labels.source', 'https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/tldr/online_45k.json'),
        bind('labels.num_train', 46_000),
        bind('batch_size', 2 * 8),
        bind('lr', 2.5e-5),
        bind('rollout_batch_size', 128),
        bind('run.seed', 1)
    )
    return locals()


def get_experiments():
    """
    policy模型训练配置
    """
    train_reward_experiments = get_train_reward_experiments()

    _books_task = combos(
        bind_nested('task', books_task),
        bind('ppo.lr', 1e-5),
        bind('ppo.total_episodes', 1_000_000),
        bind('ppo.batch_size', 512),
    )

    # 任务1. 积极情感-风格化续写
    sentiment = combos(
        _books_task,
        bind('rewards.kl_coef', 0.15),
        bind('rewards.adaptive_kl', 'on'),
        bind('rewards.adaptive_kl.target', 6.0),
        bind('rewards.train_new_model', 'on'),
        bind_nested('rewards.train_new_model', train_reward_experiments['sentiment']),
        # bind('rewards.trained_model', '/your/directory/here/reward_model/'),
        bind('run.seed', 1)
    )

    # 任务2. 物理描述-风格化续写
    descriptiveness = combos(
        _books_task,
        bind('rewards.kl_coef', 0.15),
        bind('rewards.adaptive_kl', 'on'),
        bind('rewards.adaptive_kl.target', 6.0),
        bind('rewards.train_new_model', 'on'),
        bind_nested('rewards.train_new_model', train_reward_experiments['descriptiveness']),
        # bind('rewards.trained_model', '/your/directory/here/reward_model/'),
        bind('run.seed', 1)
    )

    # 任务3. CNN/Daily Mail摘要任务
    cnndm = combos(
        bind_nested('task', summarize_cnndm_task),
        bind('rewards.train_new_model', 'on'),
        bind_nested('rewards.train_new_model', train_reward_experiments['cnndm']),
        # bind('rewards.trained_model', '/your/directory/here/reward_model/'),
        bind('ppo.total_episodes', 1_000_000),
        bind('ppo.lr', 2e-6),
        bind('rewards.kl_coef', 0.01),
        # bind('rewards.adaptive_kl', 'on'),
        # bind('rewards.adaptive_kl.target', 18.0),
        bind('ppo.batch_size', 32),
        bind('rewards.whiten', False),
        bind('run.seed', 1)
    )

    # 任务4. TL;DR摘要任务
    tldr = combos(
        bind_nested('task', summarize_tldr_task),
        bind('rewards.train_new_model', 'on'),
        bind_nested('rewards.train_new_model', train_reward_experiments['tldr']),
        # bind('rewards.trained_model', '/your/directory/here/reward_model/'),
        bind('ppo.total_episodes', 1_000_000),
        bind('ppo.lr', 2e-6),
        bind('rewards.kl_coef', 0.03), # 0.01 too low
        # bind('rewards.adaptive_kl', 'on'),
        # bind('rewards.adaptive_kl.target', 18.0),
        bind('ppo.batch_size', 32),
        bind('rewards.whiten', False),
        bind('run.seed', 1)
    )
    return locals()

# 训练策略模型
def launch_train_policy(exp, name, dry_run=False, mpi=8, mode='local', save_dir='/tmp/save/train_policy', **extra_hparams):
    """
    训练policy模型
    """
    experiment_dict = get_experiments()
    try:
        trials = experiment_dict[exp]
    except KeyError:
        raise ValueError(f"Couldn't find experiment '{exp}'")

    launch.launch_trials(
        name, 
        fn=train_policy.train, 
        trials=trials, 
        mpi=mpi, 
        mode=mode, 
        save_dir=save_dir,
        hparam_class=train_policy.HParams, 
        extra_hparams=extra_hparams, 
        dry_run=dry_run
    )


def launch_train_reward(exp, name, dry_run=False, mpi=8, mode='local', save_dir='/tmp/save/train_reward', **extra_hparams):
    """
    训练reward模型
    """
    # print(f"debug. enter launch_train_reward(). exp: {exp}, name: {name}, mpi: {mpi}, mode: {mode}, dry_run: {dry_run}")
    experiment_dict = get_train_reward_experiments()
    try:
        trials = experiment_dict[exp]
    except KeyError:
        raise ValueError(f"Couldn't find experiment '{exp}'")

    launch.launch_trials(
        name, 
        fn=train_reward.train, 
        trials=trials, 
        mpi=mpi, 
        mode=mode, 
        save_dir=save_dir,
        hparam_class=train_reward.HParams, 
        extra_hparams=extra_hparams, 
        dry_run=dry_run
    )


if __name__ == '__main__':
    launch.main(dict(
        train_policy=launch_train_policy, 
        train_reward=launch_train_reward
    ))
