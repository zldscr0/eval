import pandas as pd
import json

# 1. 读取 parquet 文件
df = pd.read_parquet('aime_2024_problems.parquet')

# 2. 写入为 jsonl 文件
with open('aime_2024_problems.jsonl', 'w') as f:
    for record in df.to_dict(orient='records'):
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
