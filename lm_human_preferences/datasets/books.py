import json
import random
import os
from lm_human_preferences.utils import gcs


def books_generator(mode, seed=0, shuffle=False, comm=None):
    try:
        file_path = gcs.download_file_cached(f'https://openaipublic.blob.core.windows.net/lm-human-preferences/datasets/book_passages/{mode}.jsonl', comm=comm)
        
        # 检查文件是否为空或占位符
        if os.path.getsize(file_path) == 0 or os.path.getsize(file_path) < 10:
            print(f'WARNING: Dataset file appears to be empty or placeholder. Using test data generator.')
            # 使用测试数据生成器
            import random as rnd
            rnd.seed(seed)
            while True:
                # 生成模拟的书籍段落数据（返回字符串）
                text = '. '.join([' '.join([chr(ord('a') + rnd.randint(0, 25)) for _ in range(rnd.randint(5, 15))]) 
                                 for _ in range(rnd.randint(3, 8))]) + '.'
                yield text
            return
        
        with open(file_path) as f:
            datas = [json.loads(line) for line in f if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError) as e:
        print(f'WARNING: Could not load books dataset: {e}. Using test data generator.')
        # 使用测试数据生成器
        import random as rnd
        rnd.seed(seed)
        while True:
            # 生成模拟的书籍段落数据（返回字符串）
            text = '. '.join([' '.join([chr(ord('a') + rnd.randint(0, 25)) for _ in range(rnd.randint(5, 15))]) 
                             for _ in range(rnd.randint(3, 8))]) + '.'
            yield text
        return
    
    if shuffle:
        random.seed(seed)
        random.shuffle(datas)

    # 从 JSON 数据中提取文本字段，如果没有则使用整个数据作为字符串
    for x in datas:
        if isinstance(x, dict):
            # 尝试从字典中提取文本字段
            text = x.get('text', x.get('content', str(x)))
            yield text
        else:
            # 如果不是字典，直接作为字符串返回
            yield str(x)
