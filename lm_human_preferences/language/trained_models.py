import copy
import os
import tensorflow as tf
from lm_human_preferences.language import encodings, model


class TrainedModel():
    """
    已训练的模型
    """
    def __init__(self, name, *, savedir=None, scope=None):
        """
        @name: 模型名称. 如124M
        @savedir: 模型保存路径
        @scope: 
        """
        self.name = name
        self.scope = scope
        if savedir is None:
            local_base = os.environ.get('GPT2_MODEL_PATH', os.path.expanduser('~/gpt-2-models'))
            local_model_path = os.path.join(local_base, 'models', name)
            print(f"local_model_path: {local_model_path}")
            # local_model_path: /root/gpt-2-models/models/124M
            
            """
            (base) root@iZ0jlfyn5du7ptefx2tr5vZ:~/PycharmProjects/lm-human-preferences# tree ~/gpt-2-models/models/124M/
            /root/gpt-2-models/models/124M/
            ├── checkpoint
            ├── hparams.json
            ├── model.ckpt.data-00000-of-00001
            ├── model.ckpt.index
            └── model.ckpt.meta
            
            model.ckpt.三个文件配套使用, 共同组成完整TF checkpoint: 
            1. model.ckpt.data-00000-of-00001
            存储模型参数(权重、偏置等数值数据)的主文件. 模型很大时通常会被分为多个data文件
            2. model.ckpt.index
            索引, 描述了权重在.data文件中的映射和位置. 没有index就无法找到和读取参数数据
            3. model.ckpt.meta
            计算图结构(ops、variable scope、操作关系等). 用来描述模型的结构和定义

            (base) root@iZ0jlfyn5du7ptefx2tr5vZ:~/gpt-2-models/models/124M# du -sh *
            4.0K    checkpoint
            4.0K    hparams.json
            475M    model.ckpt.data-00000-of-00001
            8.0K    model.ckpt.index
            464K    model.ckpt.meta
            
            (base) root@iZ0jlfyn5du7ptefx2tr5vZ:~/gpt-2-models/models/124M# cat checkpoint
            model_checkpoint_path: "model.ckpt"
            all_model_checkpoint_paths: "model.ckpt"
            
            (base) root@iZ0jlfyn5du7ptefx2tr5vZ:~/gpt-2-models/models/124M# cat hparams.json
            {
                "n_vocab": 50257,
                "n_ctx": 1024,
                "n_embd": 768,
                "n_head": 12,
                "n_layer": 12
            }
            """
            if (os.path.exists(os.path.join(local_model_path, 'hparams.json')) or os.path.exists(os.path.join(local_model_path, 'checkpoint'))):
                self.savedir = local_model_path
            else:
                self.savedir = os.path.join('gs://gpt-2/models/', name) # 回退到 GCS 路径
        else:
            self.savedir = savedir
            
        if name == 'test':
            self.encoding = encodings.Test
        else:
            self.encoding = encodings.Main
        self._hparams = None
        print(f"name: {name}, scope: {self.scope}, self.savedir: {self.savedir}, self.encoding: {self.encoding}")
        # name: 124M, scope: None, self.savedir: /root/gpt-2-models/models/124M, self.encoding: <lm_human_preferences.language.encodings.Encoding object at 0x7f60bb381f10>


    def checkpoint(self):
        if self.name == 'test':
            return None
        
        ckpt = tf.train.latest_checkpoint(self.savedir)
        print(f"self.savedir: {self.savedir}")  # self.savedir: /root/gpt-2-models/models/124M
        print(f"ckpt: {ckpt}")                  # ckpt: /root/gpt-2-models/models/124M/model.ckpt

        if ckpt is not None:
            return ckpt
        return tf.train.latest_checkpoint(os.path.join(self.savedir, 'checkpoints'))


    def hparams(self):
        """
        加载hparams对象
        """
        if self._hparams is None:
            if self.name == 'test':
                hparams = test_hparams()
            else:
                hparams = load_hparams(os.path.join(self.savedir, 'hparams.json'))
            self._hparams = hparams
        print(f"[hparams]. name: {self.name}, _hparams: {self._hparams}")
        # name: 124M, _hparams: HParams(n_vocab=50257, n_ctx=1024, n_embd=768, n_head=12, n_layer=12, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1, head_pdrop=0.1)
        return copy.deepcopy(self._hparams)


    def init_op(self, params, new_scope):
        """
        用于将checkpoint中保存的模型权重加载到当前模型权重中(即参数迁移/热启动, 模型可直接基于训练好的权重继续训练或微调)
        @params: 需要被初始化赋值的 ckpt变量名称->变量
        @new_scope: 当前policy/model变量命名空间(用于处理变量名和checkpoint的匹配)
        
        主要功能
        实现checkpoint到当前模型变量的“命名映射+类型校验+批量加载”, 是深度学习迁移、微调、RLHF 训练常用的基础设施
        兼容不同作用域下的变量名差异(新老模型scope名、不同命名风格可自动匹配)
        
        典型场景
        加载主模型参数到reward model或微调模型. 只初始化部分参数, 剩下可随机初始化
        """
        # 参数字典不为空, 否则后面操作都没意义
        assert params
        
        params = dict(**params) # 深拷贝
        checkpoint = self.checkpoint()
        available = tf.train.list_variables(checkpoint)
        
        # param形如('model/h1/attn/c_attn/w', [1, 768, 2304]), 详见: checkpoint_variable.md
        # for i, param in enumerate(available):
        #     print(f"available[{i}]: {param}")

        # 合法的可从ckpt初始化的变量名->变量对象
        unchanged = {}
        for name, shape in available:
            our_name = name
            # 处理scope不同带来的"参数名不一致"问题
            if self.scope:
                if name.startswith(self.scope):
                    our_name = name[len(self.scope):].lstrip('/')
                else:
                    continue

            # Annoying hack since some code uses 'scope/model' as the scope and other code uses just 'scope'
            our_name = f"{new_scope}/{our_name}"
            if our_name not in params:
                # NOTE: this happens for global_step and optimizer variables(e.g. beta1_power, beta2_power, blah/Adam, blah/Adam_1)
                print(f'{name} is missing for scope {new_scope}')
                continue
            var = params[our_name]
            del params[our_name]
            assert var.shape == shape, f"Shape mismatch: {var.op.name}.shape = {var.shape} != {shape}"
            unchanged[name] = var
        
        # 对剩下没有被checkpoint覆盖变量做通报, debug时能及时发现变量名字或scope命名问题
        for name in params.keys():
            print(f'Param {name} is missing from checkpoint {checkpoint}')
        """
        Param ref_policy/model/heads/value/w is missing from checkpoint /root/gpt-2-models/models/124M/model.ckpt
        Param ref_policy/model/heads/value/b is missing from checkpoint /root/gpt-2-models/models/124M/model.ckpt
        Param reward_model/model/heads/reward/w is missing from checkpoint /root/gpt-2-models/models/124M/model.ckpt
        Param reward_model/model/heads/reward/b is missing from checkpoint /root/gpt-2-models/models/124M/model.ckpt
        Param reward_model/reward_norm/gain is missing from checkpoint /root/gpt-2-models/models/124M/model.ckpt
        Param reward_model/reward_norm/bias is missing from checkpoint /root/gpt-2-models/models/124M/model.ckpt
        """
        
        # init_from_checkpoint方法把checkpoint里的参数值依次赋到当前模型变量中.
        # 该函数并不是直接“赋值”变量或返回新变量, 而是更换变量的初始化器initializer.
        # 后续在初始化发生时(sess.run(tf.global_variables_initializer())), 变量会自动从checkpoint文件里读取对应权重
        tf.train.init_from_checkpoint(checkpoint, unchanged)


def load_hparams(file):
    """
    从json文件中加载hparams对象
    """
    hparams = model.HParams()
    hparams.override_from_json_file(file)
    return hparams


def test_hparams():
    hparams = model.HParams()
    hparams.override_from_dict(dict(
        n_vocab=27,  # Corresponds to random encoding length
        n_ctx=8,
        n_layer=2,
        n_embd=7,
        n_head=1,
    ))
    return hparams
