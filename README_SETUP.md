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


```bash
(py37) root@iZ0jlfyn5du7ptefx2tr5vZ:~/PycharmProjects/lm-human-preferences# sh run.sh 
hparams:
    batch_size: 32
    debug_normalize: 0
    labels:
        num_train: 4992
        source: https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/descriptiveness/offline_5k.json
        type: best_of_4
    lr: 5e-05
    normalize_after: True
    normalize_before: True
    normalize_samples: 256
    rollout_batch_size: 512
    run:
        log_interval: 10
        save_dir: /tmp/save/train_reward/testdesc-2512311701
        save_interval: 50
        seed: 1
    task:
        start_text: .
        end_text: .
        penalty_reward_value: -1
        policy:
            initial_model: 124M
            temperature: 0.7
        query_dataset: books
        query_length: 64
        query_prefix: 
        query_suffix: 
        response_length: 24
        truncate_after: 16
        truncate_token: 13
    model_hparams:
        attn_pdrop: 0.1
        embd_pdrop: 0.1
        head_pdrop: 0.1
        n_ctx: 1024
        n_embd: 768
        n_head: 12
        n_layer: 12
        n_vocab: 50257
        resid_pdrop: 0.1

WARNING:tensorflow:From /root/.local/share/virtualenvs/lm-human-preferences-XpxZn-hG/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From /root/.local/share/virtualenvs/lm-human-preferences-XpxZn-hG/lib/python3.7/site-packages/tensorflow/python/framework/function.py:1007: calling Graph.create_op (from tensorflow.python.framework.ops) with compute_shapes is deprecated and will be removed in a future version.
Instructions for updating:
Shapes are always computed; don't use the compute_shapes as it has no effect.
Param ref_policy/model/heads/value/w is missing from checkpoint /root/gpt-2-models/models/124M/model.ckpt
Param ref_policy/model/heads/value/b is missing from checkpoint /root/gpt-2-models/models/124M/model.ckpt
Param reward_model/model/heads/reward/w is missing from checkpoint /root/gpt-2-models/models/124M/model.ckpt
Param reward_model/model/heads/reward/b is missing from checkpoint /root/gpt-2-models/models/124M/model.ckpt
Param reward_model/reward_norm/gain is missing from checkpoint /root/gpt-2-models/models/124M/model.ckpt
Param reward_model/reward_norm/bias is missing from checkpoint /root/gpt-2-models/models/124M/model.ckpt
WARNING:tensorflow:From /root/.local/share/virtualenvs/lm-human-preferences-XpxZn-hG/lib/python3.7/site-packages/tensorflow/python/data/ops/dataset_ops.py:429: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
Instructions for updating:
tf.py_func is deprecated in TF V2. Instead, use
    tf.py_function, which takes a python function which manipulates tf eager
    tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
    an ndarray (just call tensor.numpy()) but having access to eager tensors
    means `tf.py_function`s can use accelerators such as GPUs as well as
    being differentiable using a gradient tape.
    
WARNING:tensorflow:From /root/.local/share/virtualenvs/lm-human-preferences-XpxZn-hG/lib/python3.7/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
WARNING:tensorflow:From /root/.local/share/virtualenvs/lm-human-preferences-XpxZn-hG/lib/python3.7/site-packages/tensorflow/python/ops/math_grad.py:102: div (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Deprecated in favor of operator or tf.math.divide.
Will save to /tmp/save/train_reward/testdesc-2601022114
2026-01-02 21:14:46.635636: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2026-01-02 21:14:46.655681: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2699995000 Hz
2026-01-02 21:14:46.655965: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x17f12d60 executing computations on platform Host. Devices:
2026-01-02 21:14:46.655990: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
Num labels found in source: 6260
WARNING: Dataset file appears to be empty or placeholder. Using test data generator.
2026-01-02 21:16:09.646381: W tensorflow/core/framework/allocator.cc:124] Allocation of 6587285504 exceeds 10% of system memory.
2026-01-02 21:17:38.420132: W tensorflow/core/framework/allocator.cc:124] Allocation of 3019898880 exceeds 10% of system memory.
2026-01-02 21:17:42.887115: W tensorflow/core/framework/allocator.cc:124] Allocation of 3057647616 exceeds 10% of system memory.
2026-01-02 21:17:47.330919: W tensorflow/core/framework/allocator.cc:124] Allocation of 3095396352 exceeds 10% of system memory.
2026-01-02 21:17:51.807249: W tensorflow/core/framework/allocator.cc:124] Allocation of 3133145088 exceeds 10% of system memory.
targets: 0.0 +- 1.0
before normalize: 4.287679672241211 +- 2.0098273771920887
0 training on 4992 in batches of 32
k =  error , v =  1.8800657
k =  loss , v =  1.8800657
k =  error , v =  1.2473459
k =  loss , v =  1.2473459
k =  error , v =  1.088949
k =  loss , v =  1.088949
k =  error , v =  0.97328025
k =  loss , v =  0.97328025
k =  error , v =  0.9752164
k =  loss , v =  0.9752164
k =  error , v =  1.1013259
k =  loss , v =  1.1013259
k =  error , v =  1.0258518
k =  loss , v =  1.0258518
k =  error , v =  0.9876937
k =  loss , v =  0.9876937
k =  error , v =  1.1937894
k =  loss , v =  1.1937894
k =  error , v =  1.1092846
k =  loss , v =  1.1092846
k =  error , v =  0.9600097
k =  loss , v =  0.9600097
k =  error , v =  1.0529504
k =  loss , v =  1.0529504
```


```bash
(base) root@iZ0jlfyn5du7ptefx2tr5vZ:~/PycharmProjects/lm-human-preferences# tree /tmp/save/
/tmp/save/
└── train_reward
    └── testdesc-2601031229
        ├── reward_model
        │   ├── encoding
        │   └── hparams.json
        ├── tb
        │   └── reward_model
        │       └── events.out.tfevents.1767414594.iZ0jlfyn5du7ptefx2tr5vZ.v2
        └── train_reward_hparams.json
```


```
"main"(base) root@iZ0jlfyn5du7ptefx2tr5vZ:~/PycharmProjects/lm-human-preferences# cat /tmp/save/train_reward/testdesc-2601031229/reward_model/hparams.json 
{
  "n_vocab": 50257,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "embd_pdrop": 0.1,
  "attn_pdrop": 0.1,
  "resid_pdrop": 0.1,
  "head_pdrop": 0.1
}

(base) root@iZ0jlfyn5du7ptefx2tr5vZ:~/PycharmProjects/lm-human-preferences# cat /tmp/save/train_reward/testdesc-2601031229/reward_model/encoding 
"main"

(base) root@iZ0jlfyn5du7ptefx2tr5vZ:~/PycharmProjects/lm-human-preferences# cat /tmp/save/train_reward/testdesc-2601031229/train_reward_hparams.json 
{
  "run": {
    "seed": 1,
    "log_interval": 10,
    "save_interval": 50,
    "save_dir": "/tmp/save/train_reward/testdesc-2601031229"
  },
  "task": {
    "query_length": 64,
    "query_dataset": "books",
    "query_prefix": "",
    "query_suffix": "",
    "start_text": ".",
    "end_text": ".",
    "response_length": 24,
    "truncate_token": 13,
    "truncate_after": 16,
    "penalty_reward_value": -1,
    "policy": {
      "temperature": 0.7,
      "initial_model": "124M"
    }
  },
  "labels": {
    "type": "best_of_4",
    "num_train": 4992,
    "source": "https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/descriptiveness/offline_5k.json"
  },
  "batch_size": 32,
  "lr": 5e-05,
  "rollout_batch_size": 512,
  "normalize_samples": 256,
  "debug_normalize": 0,
  "normalize_before": true,
  "normalize_after": true
}
```