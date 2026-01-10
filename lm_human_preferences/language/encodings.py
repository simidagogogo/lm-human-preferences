"""
Byte pair encoding utilities
"""

import json
import os
from functools import lru_cache

import tensorflow as tf
import regex as re

# TODO
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    获取当前word中所有相邻字符对
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class ReversibleEncoder:
    """
    可逆分词器与编码器, 主要用途:
    1. 编码encode: 文本->tokenid列表
    2. 解码decode: tokenid列表->文本
    3. 基于BPE算法, 并高度兼容GPT-2/BPE分词行为
    4. 可逆性、容错性强, 具备padding/末尾标记处理
    """
    def __init__(self, encoder, bpe_merges, errors="replace", eot_token=None):
        """
        @encoder: unicode字符串 -> tokenid映射
        @bpe_merges: 词对合并规则, 每个元素都是一个二元组bigram, 配合BPE作用
        @errors: 解码时如何处理错误(如"replace"/"ignore"/"strict")
        @eot_token: 可选. end-of-text特殊token的id, 用于生成终止符
        """
        # unicode字符串->token_id
        self.encoder = encoder 
        
        # token_id->unicode字符串
        self.decoder = {v: k for k, v in self.encoder.items()}  
        
        # byte->unicode. 字节级编码, 保证二进制数据可安全转成文本.
        self.byte_encoder = bytes_to_unicode()
        
        # unicode->byte
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # how to handle errors in decoding
        self.errors = errors  
       
        # BPE合并规则的优先级映射, rank越小优先级越高
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.eot_token = eot_token
        
        # 缓存每个分词token的BPE结果
        self.cache = {}
        
        # 填充tokenid, 用于补齐序列长度
        self.padding_token = len(encoder) + 2 # +2 unnecessary, for historical reasons
        
        # padding_token的反向解码为空字符串, 保证解码时不会报错或出现脏数据
        self.decoder[self.padding_token] = ''

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        # 应该添加re.IGNORECASE, 这样BPE合并操作才能对大写形式的缩写也生效
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        """
        unicode字符串根据已知bpe_merges规则反复合并成更粗的子词单元, 用于子词tokenization
        @token: unicode字符串
        @return: 空格拼接的unicode字符串
        
        步骤
        1. 缓存命中: 已分割过, 直接查表返回
        2. 拆成tuple: 每个字符拆成元组元素, 如'test'->('t','e','s','t')
        3. 获取所有可合并pair
        4. 循环: 找到最优先可合并的pair, 根据bpe_ranks优先级合并并更新word. 直到再无新pair可合并时停止
        5. 最后将所有合并结果拼成字符串，加入缓存并返回
        这种循环BPE方式就是让token以最少步骤合成整个词表中注册过的大token
        """
        # 1. 缓存命中
        if token in self.cache:
            return self.cache[token]
        
        # 2. 拆成tuple
        word = tuple(token)
        
        # 3. 找到所有可合并pair
        pairs = get_pairs(word)

        # 单字符token直接返回
        if not pairs:
            return token

        # 4. BPE核心循环: 反复合并优先级最高的pair, 直到没有可合并pair
        while True:
            # 4.1 选择当前所有pair中“BPE优先级最高”的bigram
            bigram = min(
                pairs, 
                key=lambda pair: self.bpe_ranks.get(pair, float("inf"))
            )
            
            # 4.2 如果最优bigram没在BPE规则表里, 说明已经无法继续合并了, 退出循环
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            
            # 4.3 遍历当前的word, 执行bigram合并
            while i < len(word):
                try:
                    # 4.3.1 找到下一个first的位置，从当前位置i开始. j是first的下标
                    j = word.index(first, i)
                    
                    # 4.3.2 先把i到j之间所有字符加入new_word
                    new_word.extend(word[i:j])
                    i = j
                except:
                    # 没找到first，直接把剩下的字符全部加入new_word，退出内层循环
                    new_word.extend(word[i:])
                    break

                # 4.3.3 如果first后面正好跟着second，说明这两个可以合并（==要按照bigram合并原则合并）
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2 # 跳过已合并的两个字符
                else:
                    # 不是bigram匹配，按原样加入new_word
                    new_word.append(word[i])
                    i += 1

            # 4.4 更新word,准备下一轮bpe合并. 注意此时word已比原来短
            new_word = tuple(new_word)
            word = new_word
            
            # 4.5 如果变成单个元素, 说明已经无法继续合并(是词表中最长的token), 退出循环
            if len(word) == 1:
                break
            else:
                # 4.6 否则重新计算所有可合并pair, 进入下一轮合并
                pairs = get_pairs(word)
        
        # 5. 合并结束: 用空格拼接所有子词, 例如()'he','llo') -> 'he llo')
        word = " ".join(word)
        
        # 6. 存入缓存, 方便下次直接查表, 加速分词
        self.cache[token] = word
        return word


    def encode(self, text):
        """
        原始文本编码为tokenid
        
        在GPT、BERT等模型中, tokenizer常常先把字符串编码成UTF-8字节
        英文字符用1个字节, 特殊字符可能用2~4字节(中文、emoji)
        
        # 将字符串按UTF-8编码转换成字节流
        s = "hello世界😊"
        b = s.encode("utf-8")  # 得到字节流
        print(b) # 输出: b'hello\xe4\xb8\x96\xe7\x95\x8c\xf0\x9f\x98\x8a' 
        """
        # print(f"text: {text}")
        bpe_token_ids = []
        for token in re.findall(self.pat, text):
            # 原始文本->字节序列->unicode字符串. 
            # 保证所有token均为安全显示字符串,无论原始内容是文本还是二进制
            unicode_token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens = [bpe_token for bpe_token in self.bpe(unicode_token).split(" ")]           
            tokens = [self.encoder[bpe_token] for bpe_token in bpe_tokens]
            # print(f"token: {token}, unicode_token: {unicode_token}, bpe_tokens: {bpe_tokens}, tokens: {tokens}")
            bpe_token_ids.extend(tokens)
        # print(f"len(bpe_token_ids): {len(bpe_token_ids)}, bpe_token_ids[:10]: {bpe_token_ids[:10]}")
        return bpe_token_ids
    """
    text: b w s r j y u p o o v. w e i t l g. e q m b j d n p.
    token: b, unicode_token: b, bpe_tokens: ['b'], tokens: [65]
    token:  w, unicode_token: Ġw, bpe_tokens: ['Ġw'], tokens: [266]
    token:  s, unicode_token: Ġs, bpe_tokens: ['Ġs'], tokens: [264]
    token:  r, unicode_token: Ġr, bpe_tokens: ['Ġr'], tokens: [374]
    token:  j, unicode_token: Ġj, bpe_tokens: ['Ġj'], tokens: [474]
    token:  y, unicode_token: Ġy, bpe_tokens: ['Ġy'], tokens: [331]
    token:  u, unicode_token: Ġu, bpe_tokens: ['Ġu'], tokens: [334]
    token:  p, unicode_token: Ġp, bpe_tokens: ['Ġp'], tokens: [279]
    token:  o, unicode_token: Ġo, bpe_tokens: ['Ġo'], tokens: [267]
    token:  o, unicode_token: Ġo, bpe_tokens: ['Ġo'], tokens: [267]
    token:  v, unicode_token: Ġv, bpe_tokens: ['Ġv'], tokens: [410]
    token: ., unicode_token: ., bpe_tokens: ['.'], tokens: [13]
    token:  w, unicode_token: Ġw, bpe_tokens: ['Ġw'], tokens: [266]
    token:  e, unicode_token: Ġe, bpe_tokens: ['Ġe'], tokens: [304]
    token:  i, unicode_token: Ġi, bpe_tokens: ['Ġi'], tokens: [1312]
    token:  t, unicode_token: Ġt, bpe_tokens: ['Ġt'], tokens: [256]
    token:  l, unicode_token: Ġl, bpe_tokens: ['Ġl'], tokens: [300]
    token:  g, unicode_token: Ġg, bpe_tokens: ['Ġg'], tokens: [308]
    token: ., unicode_token: ., bpe_tokens: ['.'], tokens: [13]
    token:  e, unicode_token: Ġe, bpe_tokens: ['Ġe'], tokens: [304]
    token:  q, unicode_token: Ġq, bpe_tokens: ['Ġq'], tokens: [10662]
    token:  m, unicode_token: Ġm, bpe_tokens: ['Ġm'], tokens: [285]
    token:  b, unicode_token: Ġb, bpe_tokens: ['Ġb'], tokens: [275]
    token:  j, unicode_token: Ġj, bpe_tokens: ['Ġj'], tokens: [474]
    token:  d, unicode_token: Ġd, bpe_tokens: ['Ġd'], tokens: [288]
    token:  n, unicode_token: Ġn, bpe_tokens: ['Ġn'], tokens: [299]
    token:  p, unicode_token: Ġp, bpe_tokens: ['Ġp'], tokens: [279]
    token: ., unicode_token: ., bpe_tokens: ['.'], tokens: [13]
    len(bpe_token_ids): 28, bpe_token_ids[:10]: [65, 266, 264, 374, 474, 331, 334, 279, 267, 267]
    begin tokens: [65, 266, 264, 374, 474, 331, 334, 279, 267, 267, 410, 13, 266, 304, 1312, 256, 300, 308, 13, 304, 10662, 285, 275, 474, 288, 299, 279, 13]
    final tokens: [266, 304, 1312, 256, 300, 308, 13, 304, 10662, 285, 275, 474, 288, 299, 279, 13, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259]
    """


    def decode(self, tokens, pretty=False):
        """
        tokenid解码为原始文本
        
        直观流程举例, 假设: 
            self.decoder = {15496: 'hello', 11: ' world', 0: '!'}
            tokens = [15496, 11, 0]
        1. "".join[self.decoder[token] for token in tokens] -> 'hello world!' # 此字符串是经过byte_encoder的特殊字符, 还需一步还原
        2. self.byte_decoder[...] # 把每个“BPE生成的特殊编码的字符"还原成utf-8字节
        3. bytearray(...).decode("utf-8") #再用utf-8全面解码成最终人类可读文本
        
        为什么这么设计？
        支持任意二进制-Token-文本无损还原/模型输入/输出和真实文本/二进制之间不丢任何信息(比如emoji、汉字、特殊符号等)
        能和encode方法完全对称, 无论怎么 encode, decode 都能还原回原文. 整个编码-解码过程都是可逆和信息不丢失的！
        支持各类模型, 例如GPT-2/3、T5等, 实际就是在做“tokenizer的可逆映射”
        """
        del pretty
        
        # tokenid->unicode字符串. 备注:BPE分词结果可能会有前导空格
        text = "".join([self.decoder[token] for token in tokens])
        print(f"tokens: {tokens}")
        print(f"unicode text: {text}")
        
        # unicode字符串->byte序列->原始文本
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        print(f"final text: {text}")
        
        """
        tokens: [19638, 38271, 37400, 15425, 45738, 43003, 7420, 47041, 7291, 19744, 44978, 41022, 15421, 23106, 39383, 14100, 7453, 7780, 32145, 40536, 19163, 18020, 20907, 18273]
        unicode text: ĠpacketaucusesĠNingĠvillagesĠFrankensteinĠcanineĠSaudibahĠfacilitiesĠpricedanskiĠMbpsĠtraditionsHeightĠdedicatemillionĠeligermĠDoddĠComplianceĠappreciationĠCaribbeanĠPTSUntil
        final text: packetaucuses Ning villages Frankenstein canine Saudibah facilities pricedanski Mbps traditionsHeight dedicatemillion eligerm Dodd Compliance appreciation Caribbean PTSUntil
        packetaucuses Ning villages Frankenstein canine Saudibah facilities pricedanski Mbps traditionsHeight dedicatemillion eligerm Dodd Compliance appreciation Caribbean PTSUntil        
        """
        return text


def read_file(path):
    """
    以二进制方式读取文件内容
    """
    with tf.gfile.Open(path, "rb") as fh:
        return fh.read()


class Encoding:
    def __init__(
        self,
        name,
        *,
        n_vocab=0,
        eot_token=None,
        encoder_path="encoder.json",
        bpe_path="vocab.bpe",
        base_path=None,
    ):
        self.name = name
        self.eot_token = eot_token
        self.n_vocab = n_vocab

        """
         ~/PycharmProjects/lm-human-preferences master ± tree ~/gpt-2-models/encodings 
        /Users/xiangqian/gpt-2-models/encodings
        └── main
            ├── encoder.json
            └── vocab.bpe
        """
        if base_path is None:
            local_base = os.environ.get('GPT2_MODEL_PATH', os.path.expanduser('~/gpt-2-models'))
            local_encoding_path = os.path.join(local_base, 'encodings', name)
            if os.path.exists(os.path.join(local_encoding_path, encoder_path)):
                base_path = local_encoding_path
            else:
                base_path = os.path.join("gs://gpt-2/encodings", name) # 回退到 GCS 路径

        self.base_path = base_path
        print(f"name: {name}, self.base_path: {self.base_path}")
        if name != "test":
            self.encoder_path = os.path.join(self.base_path, encoder_path)
            self.bpe_path = os.path.join(self.base_path, bpe_path)
            print(f"name: {name}, self.encoder_path: {self.encoder_path}")
            print(f"name: {name}, self.bpe_path: {self.bpe_path}")
        """
        name: main, self.base_path: /root/gpt-2-models/encodings/main
        name: main, self.encoder_path: /root/gpt-2-models/encodings/main/encoder.json
        name: main, self.bpe_path: /root/gpt-2-models/encodings/main/vocab.bpe
        name: test, self.base_path: gs://gpt-2/encodings/test
        """

    def get_encoder(self):
        if self.name == "test":
            vocab = "abcdefghijklmnopqrstuvwxyz."
            assert len(vocab) == self.n_vocab
            class TestEncoder(ReversibleEncoder):
                def __init__(self):
                    super().__init__(encoder={w: i for i, w in enumerate(vocab)}, bpe_merges=list())
                    self.padding_token = len(vocab)
                def encode(self, text):
                    return [self.encoder.get(x, len(vocab) - 1) for x in text]
                def decode(self, tokens, pretty=False):
                    return ''.join([self.decoder.get(t, '<unk>') for t in tokens])
            return TestEncoder()

        """
        典型工作流
        1. 始终用"rb"二进制读取文件, 获得最原始的数据
        2. 如果确定是文本再用正确的.decode()明确定制编码
            例如 .decode("utf-8")、.decode("gbk") 、.decode(errors="ignore") 等
        3. 这样的模式是高质量NLP工程和深度学习工程的标准实践。
        
        # 直接用文本方式打开, 有风险
        with open("xxx.json", "r") as f:
            data = f.read()  # 如果该文件不是UTF-8或者有非标准字符, 容易报错、内容读错

        # 推荐方式
        with open("xxx.json", "rb") as f:
            data = f.read()      # 得到bytes, 最安全
            text = data.decode() # 明确指定编码如utf-8
        """
        # unicode字符串->token_id. 例如{"\u0120gazed": 50255, "<|endoftext|>": 50256}
        encoder_dict = json.loads(read_file(self.encoder_path).decode())
        assert len(encoder_dict) == self.n_vocab
        
        # BPE中字符merge规则. 例如[("Ġ", "t"), ("Ġ", "a")]
        bpe_data = read_file(self.bpe_path).decode()        
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
        
        print(f"len(encoder_dict): {len(encoder_dict)}")
        print(f"encoder_dict[:5]: {dict(list(encoder_dict.items())[:5])}")
        print(f"len(bpe_merges): {len(bpe_merges)}")
        print(f"bpe_merges[:5]: {bpe_merges[:5]}")
        # len(encoder_dict): 50257
        # encoder_dict[:5]: {'!': 0, '"': 1, '#': 2, '$': 3, '%': 4}
        # len(bpe_merges): 50000
        # bpe_merges[:5]: [('Ġ', 't'), ('Ġ', 'a'), ('h', 'e'), ('i', 'n'), ('r', 'e')]
        
        encoder = ReversibleEncoder(
            encoder=encoder_dict, 
            bpe_merges=bpe_merges, 
            eot_token=self.eot_token
        )
        assert encoder.padding_token >= self.n_vocab
        return encoder


Main = Encoding("main", n_vocab=50257, eot_token=50256)
Test = Encoding("test", n_vocab=27, eot_token=26)
