import copy
import os
import tensorflow as tf
from lm_human_preferences.language import encodings, model


class TrainedModel():
    """
    已训练的模型
    """
    def __init__(self, name, *, savedir=None, scope=None):
        self.name = name
        self.scope = scope
        
        if savedir is None:
            # 优先使用本地路径（如果设置了环境变量或文件存在）
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
            """
            if (os.path.exists(os.path.join(local_model_path, 'hparams.json')) or os.path.exists(os.path.join(local_model_path, 'checkpoint'))):
                self.savedir = local_model_path
            else:
                # 回退到 GCS 路径
                self.savedir = os.path.join('gs://gpt-2/models/', name)
        else:
            self.savedir = savedir
            
        if name == 'test':
            self.encoding = encodings.Test
        else:
            self.encoding = encodings.Main
        self._hparams = None

    def checkpoint(self):
        if self.name == 'test':
            return None
        
        ckpt = tf.train.latest_checkpoint(self.savedir)
        print(f"self.savedir: {self.savedir}, ckpt: {ckpt}")
        # self.savedir: /root/gpt-2-models/models/124M, ckpt: /root/gpt-2-models/models/124M/model.ckpt
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
        return copy.deepcopy(self._hparams)

    def init_op(self, params, new_scope):
        assert params
        params = dict(**params)
        checkpoint = self.checkpoint()
        available = tf.train.list_variables(checkpoint)
        unchanged = {}

        for name, shape in available:
            our_name = name
            if self.scope:
                if name.startswith(self.scope):
                    our_name = name[len(self.scope):].lstrip('/')
                else:
                    continue
            # Annoying hack since some code uses 'scope/model' as the scope and other code uses just 'scope'
            our_name = '%s/%s' % (new_scope, our_name)
            if our_name not in params:
                # NOTE: this happens for global_step and optimizer variables
                # (e.g. beta1_power, beta2_power, blah/Adam, blah/Adam_1)
                # print(f'{name} is missing for scope {new_scope}')
                continue
            var = params[our_name]
            del params[our_name]
            assert var.shape == shape, 'Shape mismatch: %s.shape = %s != %s' % (var.op.name, var.shape, shape)
            unchanged[name] = var
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
