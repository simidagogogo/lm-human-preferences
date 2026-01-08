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

```bash
text: s o q t v. p q v b h. e o t a v. p m q n h w s h o x g o e. h a n t b. b g a k f g h e v y r i t l. v g x w i y q j e v q h x., tokens: [82, 267, 10662, 256, 410, 13, 279, 10662, 410, 275, 289, 13, 304, 267, 256, 257, 410, 13, 279, 285, 10662, 299, 289, 266, 264, 289, 267, 2124, 308, 267, 304, 13, 289, 257, 299, 256, 275, 13, 275, 308, 257, 479, 277, 308, 289, 304, 410, 331, 374, 1312, 256, 300, 13, 410, 308, 2124, 266, 1312, 331, 10662, 474, 304, 410, 10662, 289, 2124, 13], len(tokens): 67
start_token: 13
end_token: 13
text: l o l w s t i s e s. f d i n m s q l y u j f t z. n h n u v m r c u l. u n l h r r c t h r. e p d o s n i b p z n c s v. s q z p m t v g d i z j p. c e r m s w l n y. z i j e n p t v y w e n b v., tokens: [75, 267, 300, 266, 264, 256, 1312, 264, 304, 264, 13, 277, 288, 1312, 299, 285, 264, 10662, 300, 331, 334, 474, 277, 256, 1976, 13, 299, 289, 299, 334, 410, 285, 374, 269, 334, 300, 13, 334, 299, 300, 289, 374, 374, 269, 256, 289, 374, 13, 304, 279, 288, 267, 264, 299, 1312, 275, 279, 1976, 299, 269, 264, 410, 13, 264, 10662, 1976, 279, 285, 256, 410, 308, 288, 1312, 1976, 474, 279, 13, 269, 304, 374, 285, 264, 266, 300, 299, 331, 13, 1976, 1312, 474, 304, 299, 279, 256, 410, 331, 266, 304, 299, 275, 410, 13], len(tokens): 102
start_token: 13
end_token: 13
text: s u h m q t y r u v. o r p a y t j j i f h b r. n c y w n l. f o n e g. w f q i z. j u h n b t c u y e w d. g e j p l. e z n z c o z r n a l k., tokens: [82, 334, 289, 285, 10662, 256, 331, 374, 334, 410, 13, 267, 374, 279, 257, 331, 256, 474, 474, 1312, 277, 289, 275, 374, 13, 299, 269, 331, 266, 299, 300, 13, 277, 267, 299, 304, 308, 13, 266, 277, 10662, 1312, 1976, 13, 474, 334, 289, 299, 275, 256, 269, 334, 331, 304, 266, 288, 13, 308, 304, 474, 279, 300, 13, 304, 1976, 299, 1976, 269, 267, 1976, 374, 299, 257, 300, 479, 13], len(tokens): 76
start_token: 13
end_token: 13
text: h w h u h m b c. k e e l g g c x m n n. i v h z u i u a w u. m u f t g i c w q w. j p u c j. h m m l b b r h o l j t. s n u i a q h k v u c z m m. u j q o n z p c p p j f c n z., tokens: [71, 266, 289, 334, 289, 285, 275, 269, 13, 479, 304, 304, 300, 308, 308, 269, 2124, 285, 299, 299, 13, 1312, 410, 289, 1976, 334, 1312, 334, 257, 266, 334, 13, 285, 334, 277, 256, 308, 1312, 269, 266, 10662, 266, 13, 474, 279, 334, 269, 474, 13, 289, 285, 285, 300, 275, 275, 374, 289, 267, 300, 474, 256, 13, 264, 299, 334, 1312, 257, 10662, 289, 479, 410, 334, 269, 1976, 285, 285, 13, 334, 474, 10662, 267, 299, 1976, 279, 269, 279, 279, 474, 277, 269, 299, 1976, 13], len(tokens): 93
start_token: 13
end_token: 13
text: t g d q i v m w. q e a o d e. v m o z e y i r c e a l s m., tokens: [83, 308, 288, 10662, 1312, 410, 285, 266, 13, 10662, 304, 257, 267, 288, 304, 13, 410, 285, 267, 1976, 304, 331, 1312, 374, 269, 304, 257, 300, 264, 285, 13], len(tokens): 31
start_token: 13
end_token: 13
text: h w a h g r i i c h h. z b m j m q. d s x u p b p., tokens: [71, 266, 257, 289, 308, 374, 1312, 1312, 269, 289, 289, 13, 1976, 275, 285, 474, 285, 10662, 13, 288, 264, 2124, 334, 279, 275, 279, 13], len(tokens): 27
start_token: 13
end_token: 13
text: v s p r b v z s. t o s d u w c i v j e. x z o o m a q t z e h f., tokens: [85, 264, 279, 374, 275, 410, 1976, 264, 13, 256, 267, 264, 288, 334, 266, 269, 1312, 410, 474, 304, 13, 2124, 1976, 267, 267, 285, 257, 10662, 256, 1976, 304, 289, 277, 13], len(tokens): 34
start_token: 13
end_token: 13
text: g x m z r v q v z c t b i. l j g i g c s z g z b h m. n z e w i c j d k. q c a n q y q z l e a t l., tokens: [70, 2124, 285, 1976, 374, 410, 10662, 410, 1976, 269, 256, 275, 1312, 13, 300, 474, 308, 1312, 308, 269, 264, 1976, 308, 1976, 275, 289, 285, 13, 299, 1976, 304, 266, 1312, 269, 474, 288, 479, 13, 10662, 269, 257, 299, 10662, 331, 10662, 1976, 300, 304, 257, 256, 300, 13], len(tokens): 52
start_token: 13
end_token: 13
text: l e u o u w b x z d t. f l s d q q f v e u r. p c p w b. j q p b z a a b t v. o i a b x u. f w b i g y., tokens: [75, 304, 334, 267, 334, 266, 275, 2124, 1976, 288, 256, 13, 277, 300, 264, 288, 10662, 10662, 277, 410, 304, 334, 374, 13, 279, 269, 279, 266, 275, 13, 474, 10662, 279, 275, 1976, 257, 257, 275, 256, 410, 13, 267, 1312, 257, 275, 2124, 334, 13, 277, 266, 275, 1312, 308, 331, 13], len(tokens): 55
start_token: 13
end_token: 13
text: i i i l h b. q w w y r j g i n. v c i l g. b s t u u z h. h o s j g v g s q v a u g. p c u h t c x z d a z v r q. c h v k c n z c q r. w y n q g q., tokens: [72, 1312, 1312, 300, 289, 275, 13, 10662, 266, 266, 331, 374, 474, 308, 1312, 299, 13, 410, 269, 1312, 300, 308, 13, 275, 264, 256, 334, 334, 1976, 289, 13, 289, 267, 264, 474, 308, 410, 308, 264, 10662, 410, 257, 334, 308, 13, 279, 269, 334, 289, 256, 269, 2124, 1976, 288, 257, 1976, 410, 374, 10662, 13, 269, 289, 410, 479, 269, 299, 1976, 269, 10662, 374, 13, 266, 331, 299, 10662, 308, 10662, 13], len(tokens): 78
start_token: 13
end_token: 13
text: v e g a p h n k g s j t c d. d c t x d v j. b q t j b q m v w b r j b. i u o f y k s u z u j a. n u r m y c k c c g n x v., tokens: [85, 304, 308, 257, 279, 289, 299, 479, 308, 264, 474, 256, 269, 288, 13, 288, 269, 256, 2124, 288, 410, 474, 13, 275, 10662, 256, 474, 275, 10662, 285, 410, 266, 275, 374, 474, 275, 13, 1312, 334, 267, 277, 331, 479, 264, 334, 1976, 334, 474, 257, 13, 299, 334, 374, 285, 331, 269, 479, 269, 269, 308, 299, 2124, 410, 13], len(tokens): 64
start_token: 13
end_token: 13
text: c d b s f h k. u c v n y n. p n p p t. g r t f r w f., tokens: [66, 288, 275, 264, 277, 289, 479, 13, 334, 269, 410, 299, 331, 299, 13, 279, 299, 279, 279, 256, 13, 308, 374, 256, 277, 374, 266, 277, 13], len(tokens): 29
start_token: 13
end_token: 13
text: e l v l t j v l s l. t i z f x q d i t. j q g j o r b o g p s m. l e n l b b g j a p s d., tokens: [68, 300, 410, 300, 256, 474, 410, 300, 264, 300, 13, 256, 1312, 1976, 277, 2124, 10662, 288, 1312, 256, 13, 474, 10662, 308, 474, 267, 374, 275, 267, 308, 279, 264, 285, 13, 300, 304, 299, 300, 275, 275, 308, 474, 257, 279, 264, 288, 13], len(tokens): 47
start_token: 13
end_token: 13
text: y d x r e a k x s c n. f r l t l x e n v a y a. s f d c x f y s s e i l a e l. q e p y r o y z p., tokens: [88, 288, 2124, 374, 304, 257, 479, 2124, 264, 269, 299, 13, 277, 374, 300, 256, 300, 2124, 304, 299, 410, 257, 331, 257, 13, 264, 277, 288, 269, 2124, 277, 331, 264, 264, 304, 1312, 300, 257, 304, 300, 13, 10662, 304, 279, 331, 374, 267, 331, 1976, 279, 13], len(tokens): 51
start_token: 13
end_token: 13
text: t u s k q l y l t o n a. d j c u h l a c p a i t. e h o p c u l d. z c s u x u n. q e o t b m d b z j r d., tokens: [83, 334, 264, 479, 10662, 300, 331, 300, 256, 267, 299, 257, 13, 288, 474, 269, 334, 289, 300, 257, 269, 279, 257, 1312, 256, 13, 304, 289, 267, 279, 269, 334, 300, 288, 13, 1976, 269, 264, 334, 2124, 334, 299, 13, 10662, 304, 267, 256, 275, 285, 288, 275, 1976, 474, 374, 288, 13], len(tokens): 56
start_token: 13
end_token: 13
text: p z z f i. i v w u z g j. o u w l m c b p t d. b a c d w j l w o k f t., tokens: [79, 1976, 1976, 277, 1312, 13, 1312, 410, 266, 334, 1976, 308, 474, 13, 267, 334, 266, 300, 285, 269, 275, 279, 256, 288, 13, 275, 257, 269, 288, 266, 474, 300, 266, 267, 479, 277, 256, 13], len(tokens): 38
start_token: 13
end_token: 13
text: k v h f i h g c e c v a. c q g c q y j a w m e p j d u. e q w w q z c., tokens: [74, 410, 289, 277, 1312, 289, 308, 269, 304, 269, 410, 257, 13, 269, 10662, 308, 269, 10662, 331, 474, 257, 266, 285, 304, 279, 474, 288, 334, 13, 304, 10662, 266, 266, 10662, 1976, 269, 13], len(tokens): 37
start_token: 13
end_token: 13
text: u o k r v f j a b q n a q o. d b h g d v z b l v w r d d g. s n c l d b h r m r a c r. b f t h l v w m t q s. j r s x q j h l r c h. b p d s v y v j b m x. u b a k c a q v o m w j., tokens: [84, 267, 479, 374, 410, 277, 474, 257, 275, 10662, 299, 257, 10662, 267, 13, 288, 275, 289, 308, 288, 410, 1976, 275, 300, 410, 266, 374, 288, 288, 308, 13, 264, 299, 269, 300, 288, 275, 289, 374, 285, 374, 257, 269, 374, 13, 275, 277, 256, 289, 300, 410, 266, 285, 256, 10662, 264, 13, 474, 374, 264, 2124, 10662, 474, 289, 300, 374, 269, 289, 13, 275, 279, 288, 264, 410, 331, 410, 474, 275, 285, 2124, 13, 334, 275, 257, 479, 269, 257, 10662, 410, 267, 285, 266, 474, 13], len(tokens): 94
start_token: 13
end_token: 13
text: o m v b a n r d y s e h o i s. n v r q c. r k l w w y i w v f. q x b g q e g e a x n r. r r v p g h y l b z o q k q a. t f p v n l r s k m l q., tokens: [78, 285, 410, 275, 257, 299, 374, 288, 331, 264, 304, 289, 267, 1312, 264, 13, 299, 410, 374, 10662, 269, 13, 374, 479, 300, 266, 266, 331, 1312, 266, 410, 277, 13, 10662, 2124, 275, 308, 10662, 304, 308, 304, 257, 2124, 299, 374, 13, 374, 374, 410, 279, 308, 289, 331, 300, 275, 1976, 267, 10662, 479, 10662, 257, 13, 256, 277, 279, 410, 299, 300, 374, 264, 479, 285, 300, 10662, 13], len(tokens): 75
start_token: 13
end_token: 13
text: q g z k g v d a a z m u g z. m o f h y z y f k o s c. v g u x o y j e t. t s d r c h k k c i n., tokens: [80, 308, 1976, 479, 308, 410, 288, 257, 257, 1976, 285, 334, 308, 1976, 13, 285, 267, 277, 289, 331, 1976, 331, 277, 479, 267, 264, 269, 13, 410, 308, 334, 2124, 267, 331, 474, 304, 256, 13, 256, 264, 288, 374, 269, 289, 479, 479, 269, 1312, 299, 13], len(tokens): 50
start_token: 13
end_token: 13
text: y c l f y a j v q f z e. g r h a t y p. f h i a d e a. a q r k b t a y u m s b y j. c f n d d t p f o v h s. d x a w r x d k a t o m g., tokens: [88, 269, 300, 277, 331, 257, 474, 410, 10662, 277, 1976, 304, 13, 308, 374, 289, 257, 256, 331, 279, 13, 277, 289, 1312, 257, 288, 304, 257, 13, 257, 10662, 374, 479, 275, 256, 257, 331, 334, 285, 264, 275, 331, 474, 13, 269, 277, 299, 288, 288, 256, 279, 277, 267, 410, 289, 264, 13, 288, 2124, 257, 266, 374, 2124, 288, 479, 257, 256, 267, 285, 308, 13], len(tokens): 71
start_token: 13
end_token: 13
text: e t h z p f v p z w. h e c w v. h o x j r g. w e t d l h k s x b u u f v. c h r b y o h. m a q e n b q n. x j a v h e x y b t d v w. h s o s e p s m c e q y t c m., tokens: [68, 256, 289, 1976, 279, 277, 410, 279, 1976, 266, 13, 289, 304, 269, 266, 410, 13, 289, 267, 2124, 474, 374, 308, 13, 266, 304, 256, 288, 300, 289, 479, 264, 2124, 275, 334, 334, 277, 410, 13, 269, 289, 374, 275, 331, 267, 289, 13, 285, 257, 10662, 304, 299, 275, 10662, 299, 13, 2124, 474, 257, 410, 289, 304, 2124, 331, 275, 256, 288, 410, 266, 13, 289, 264, 267, 264, 304, 279, 264, 285, 269, 304, 10662, 331, 256, 269, 285, 13], len(tokens): 86
start_token: 13
end_token: 13
text: v t g k g x i p f a i q j a z. h q x y f p h f u e s. u u f i b p r j l. e e b u n v w j d g t e j. t d r t g s i s k s f i l e e., tokens: [85, 256, 308, 479, 308, 2124, 1312, 279, 277, 257, 1312, 10662, 474, 257, 1976, 13, 289, 10662, 2124, 331, 277, 279, 289, 277, 334, 304, 264, 13, 334, 334, 277, 1312, 275, 279, 374, 474, 300, 13, 304, 304, 275, 334, 299, 410, 266, 474, 288, 308, 256, 304, 474, 13, 256, 288, 374, 256, 308, 264, 1312, 264, 479, 264, 277, 1312, 300, 304, 304, 13], len(tokens): 68
start_token: 13
end_token: 13
text: q r r o w h g j j t e m. l s z h g g f h c m h d e e. m q a v h w. s f u y g h i b. n n u a g h v w z m c e o x., tokens: [80, 374, 374, 267, 266, 289, 308, 474, 474, 256, 304, 285, 13, 300, 264, 1976, 289, 308, 308, 277, 289, 269, 285, 289, 288, 304, 304, 13, 285, 10662, 257, 410, 289, 266, 13, 264, 277, 334, 331, 308, 289, 1312, 275, 13, 299, 299, 334, 257, 308, 289, 410, 266, 1976, 285, 269, 304, 267, 2124, 13], len(tokens): 59
start_token: 13
end_token: 13
text: b a h r g j m a f h h. i i h v k x m q m l g j r n l. g x d g t h j t r u., tokens: [65, 257, 289, 374, 308, 474, 285, 257, 277, 289, 289, 13, 1312, 1312, 289, 410, 479, 2124, 285, 10662, 285, 300, 308, 474, 374, 299, 300, 13, 308, 2124, 288, 308, 256, 289, 474, 256, 374, 334, 13], len(tokens): 39
start_token: 13
end_token: 13
text: y i t a y g c l. q c w x d m i s d. o m a l j y d k d n. v d j w c p., tokens: [88, 1312, 256, 257, 331, 308, 269, 300, 13, 10662, 269, 266, 2124, 288, 285, 1312, 264, 288, 13, 267, 285, 257, 300, 474, 331, 288, 479, 288, 299, 13, 410, 288, 474, 266, 269, 279, 13], len(tokens): 37
start_token: 13
end_token: 13
text: m u j v u t g e c f u j k. s v l f z u v a k u x b. n x z c j l m s m c y g. r p w d p g e. w f r n i., tokens: [76, 334, 474, 410, 334, 256, 308, 304, 269, 277, 334, 474, 479, 13, 264, 410, 300, 277, 1976, 334, 410, 257, 479, 334, 2124, 275, 13, 299, 2124, 1976, 269, 474, 300, 285, 264, 285, 269, 331, 308, 13, 374, 279, 266, 288, 279, 308, 304, 13, 266, 277, 374, 299, 1312, 13], len(tokens): 54
start_token: 13
end_token: 13
text: s x z z g t e o d d w. c a s g v o m p v. s t c d j y j l k k b o q., tokens: [82, 2124, 1976, 1976, 308, 256, 304, 267, 288, 288, 266, 13, 269, 257, 264, 308, 410, 267, 285, 279, 410, 13, 264, 256, 269, 288, 474, 331, 474, 300, 479, 479, 275, 267, 10662, 13], len(tokens): 36
start_token: 13
end_token: 13
text: c f g e i t z r. n e v x d f. d e q x y m d z b s i. x m g b o v i p u c w x o. g m m o n h q g o j w l. t l m k h q k., tokens: [66, 277, 308, 304, 1312, 256, 1976, 374, 13, 299, 304, 410, 2124, 288, 277, 13, 288, 304, 10662, 2124, 331, 285, 288, 1976, 275, 264, 1312, 13, 2124, 285, 308, 275, 267, 410, 1312, 279, 334, 269, 266, 2124, 267, 13, 308, 285, 285, 267, 299, 289, 10662, 308, 267, 474, 266, 300, 13, 256, 300, 285, 479, 289, 10662, 479, 13], len(tokens): 63
start_token: 13
end_token: 13
text: e y r d e c r. e k v o t e f. b w m x g s s j n z b x s t. b m c u o e f. r k f i g v., tokens: [68, 331, 374, 288, 304, 269, 374, 13, 304, 479, 410, 267, 256, 304, 277, 13, 275, 266, 285, 2124, 308, 264, 264, 474, 299, 1976, 275, 2124, 264, 256, 13, 275, 285, 269, 334, 267, 304, 277, 13, 374, 479, 277, 1312, 308, 410, 13], len(tokens): 46
start_token: 13
end_token: 13
text: g y z w j. f p z e m h. e x k x b c p m., tokens: [70, 331, 1976, 266, 474, 13, 277, 279, 1976, 304, 285, 289, 13, 304, 2124, 479, 2124, 275, 269, 279, 285, 13], len(tokens): 22
start_token: 13
end_token: 13
text: v p j y t j c v x y n x n. o d k u z g u v u. k d w b m j p i z r d v b., tokens: [85, 279, 474, 331, 256, 474, 269, 410, 2124, 331, 299, 2124, 299, 13, 267, 288, 479, 334, 1976, 308, 334, 410, 334, 13, 479, 288, 266, 275, 285, 474, 279, 1312, 1976, 374, 288, 410, 275, 13], len(tokens): 38
start_token: 13
end_token: 13
text: h h b i p r. f p g j g z k u. b b a l t h d., tokens: [71, 289, 275, 1312, 279, 374, 13, 277, 279, 308, 474, 308, 1976, 479, 334, 13, 275, 275, 257, 300, 256, 289, 288, 13], len(tokens): 24
start_token: 13
end_token: 13
text: s l m i f p j g. y m p t c. o m k w x. o x f f t i. c c a l e u a j y p r k e. s w x g h. v d f o j x e f f w. j m u p p., tokens: [82, 300, 285, 1312, 277, 279, 474, 308, 13, 331, 285, 279, 256, 269, 13, 267, 285, 479, 266, 2124, 13, 267, 2124, 277, 277, 256, 1312, 13, 269, 269, 257, 300, 304, 334, 257, 474, 331, 279, 374, 479, 304, 13, 264, 266, 2124, 308, 289, 13, 410, 288, 277, 267, 474, 2124, 304, 277, 277, 266, 13, 474, 285, 334, 279, 279, 13], len(tokens): 65
start_token: 13
end_token: 13
text: b q k m n i t. x w e m w q. t h x o c. z d z d m k q m b t., tokens: [65, 10662, 479, 285, 299, 1312, 256, 13, 2124, 266, 304, 285, 266, 10662, 13, 256, 289, 2124, 267, 269, 13, 1976, 288, 1976, 288, 285, 479, 10662, 285, 275, 256, 13], len(tokens): 32
start_token: 13
end_token: 13
text: q a a m a k e j m n s q j. j t j c h c. s e k w s v e s h j s f m x. z c z h g o f x w f g d f h k. t k e y j. q r h x k g j. v k n p a j e t c e v. k c v g h l i v a k d., tokens: [80, 257, 257, 285, 257, 479, 304, 474, 285, 299, 264, 10662, 474, 13, 474, 256, 474, 269, 289, 269, 13, 264, 304, 479, 266, 264, 410, 304, 264, 289, 474, 264, 277, 285, 2124, 13, 1976, 269, 1976, 289, 308, 267, 277, 2124, 266, 277, 308, 288, 277, 289, 479, 13, 256, 479, 304, 331, 474, 13, 10662, 374, 289, 2124, 479, 308, 474, 13, 410, 479, 299, 279, 257, 474, 304, 256, 269, 304, 410, 13, 479, 269, 410, 308, 289, 300, 1312, 410, 257, 479, 288, 13], len(tokens): 90
start_token: 13
end_token: 13
text: d c r r o l p d f s e p. f t z n n d q i. d u l r z z j g f k g g i r. o l h j j. a x p h z c k., tokens: [67, 269, 374, 374, 267, 300, 279, 288, 277, 264, 304, 279, 13, 277, 256, 1976, 299, 299, 288, 10662, 1312, 13, 288, 334, 300, 374, 1976, 1976, 474, 308, 277, 479, 308, 308, 1312, 374, 13, 267, 300, 289, 474, 474, 13, 257, 2124, 279, 289, 1976, 269, 479, 13], len(tokens): 51
start_token: 13
end_token: 13
text: s s h f z g m. w r m z l t w k p p a b. j p l m c a. d s d k r a l a. r b g c l s k f n l. i q z g u l x t l l d. t n i k d h n a. o l x j i r z p k f j j h., tokens: [82, 264, 289, 277, 1976, 308, 285, 13, 266, 374, 285, 1976, 300, 256, 266, 479, 279, 279, 257, 275, 13, 474, 279, 300, 285, 269, 257, 13, 288, 264, 288, 479, 374, 257, 300, 257, 13, 374, 275, 308, 269, 300, 264, 479, 277, 299, 300, 13, 1312, 10662, 1976, 308, 334, 300, 2124, 256, 300, 300, 288, 13, 256, 299, 1312, 479, 288, 289, 299, 257, 13, 267, 300, 2124, 474, 1312, 374, 1976, 279, 479, 277, 474, 474, 289, 13], len(tokens): 83
start_token: 13
end_token: 13
```