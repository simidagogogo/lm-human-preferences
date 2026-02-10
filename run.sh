#!/bin/bash
# 运行脚本 - 设置必要的环境变量并运行项目

# 设置protobuf使用纯Python实现（避免版本兼容性问题）
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 设置 GPT-2 模型本地路径
export GPT2_MODEL_PATH=~/gpt-2-models

# 1. train_reward
experiment=descriptiveness
reward_experiment_name=testdesc-$(date +%y%m%d%H%M)
pipenv run ./launch.py train_reward $experiment $reward_experiment_name --mpi 1

# 2. train_policy
# trained_reward_model=/tmp/save/train_reward/$reward_experiment_name
# experiment=descriptiveness
# policy_experiment_name=testdesc-$(date +%y%m%d%H%M)
# pipenv run ./launch.py train_policy $experiment $policy_experiment_name  --mpi 1 \
# --rewards.trained_model $trained_reward_model \
# --rewards.train_new_model 'off'

# 3. 同时运行两步
# experiment=descriptiveness
# experiment_name=testdesc-$(date +%y%m%d%H%M)
# pipenv run ./launch.py train_policy $experiment $experiment_name --mpi 1

# 4. 采样
# save_dir=/tmp/save/train_policy/$policy_experiment_name
# pipenv run ./sample.py sample --save_dir $save_dir --savescope policy