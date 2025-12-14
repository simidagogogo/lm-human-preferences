# 项目配置说明

## 已完成的配置

### 1. 依赖安装
- ✅ 安装了所有必要的 Python 依赖包
- ✅ 配置了 TensorFlow 1.13.1（CPU 版本）
- ✅ 解决了 protobuf 版本兼容性问题
- ✅ 创建了 MPI mock 模块以支持单进程运行

### 2. 模型文件下载
- ✅ 已下载 GPT-2 124M 模型文件到 `~/gpt-2-models/`
- ✅ 包含以下文件：
  - `encodings/main/encoder.json` - 编码器文件
  - `encodings/main/vocab.bpe` - BPE 词汇表
  - `models/124M/hparams.json` - 模型超参数
  - `models/124M/model.ckpt.*` - 模型检查点文件
  - `models/124M/checkpoint` - 检查点索引

### 3. 代码修改
- ✅ 修改了 `encodings.py` 以支持本地路径
- ✅ 修改了 `trained_models.py` 以支持本地路径
- ✅ 修改了 `launch.py` 中的 MPI 处理以支持单进程运行
- ✅ 创建了 `run.sh` 脚本自动设置环境变量

## 使用方法

### 运行训练奖励模型
```bash
./run.sh
```

或者手动运行：
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export GPT2_MODEL_PATH=~/gpt-2-models
pipenv run ./launch.py train_reward descriptiveness testdesc-$(date +%y%m%d%H%M) --mpi 1
```

### 运行训练策略模型
```bash
# 首先需要训练奖励模型，然后：
trained_reward_model=/tmp/save/train_reward/testdesc-XXXXXX
pipenv run ./launch.py train_policy descriptiveness testdesc-$(date +%y%m%d%H%M) --mpi 1 \
  --rewards.trained_model $trained_reward_model \
  --rewards.train_new_model 'off'
```

### 从已训练的模型采样
```bash
save_dir=/tmp/save/train_policy/testdesc-XXXXXX
pipenv run ./sample.py sample --save_dir $save_dir --savescope policy --mpi 1
```

## 环境变量说明

- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`: 使用纯 Python 实现 protobuf，避免版本兼容性问题
- `GPT2_MODEL_PATH=~/gpt-2-models`: 指定 GPT-2 模型文件的本地路径

## 文件位置

- 模型文件: `~/gpt-2-models/`
- 训练输出: `/tmp/save/train_reward/` 和 `/tmp/save/train_policy/`
- 下载脚本: `download_gpt2.py`

## 注意事项

1. 项目使用 TensorFlow 1.13.1，这是一个较旧的版本
2. 在 Mac 上使用 CPU 版本的 TensorFlow，训练速度会较慢
3. MPI 功能已通过 mock 模块模拟，支持单进程运行
4. 如果遇到网络问题，模型文件已下载到本地，无需网络访问

## 重新下载模型文件

如果需要重新下载模型文件，运行：
```bash
pipenv run python download_gpt2.py
```
