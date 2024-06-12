import json
from collections import defaultdict, Counter


snowflake_score = []
with open('result_snowflake.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        snowflake_score.append(json.loads(line))

ali_score = []
with open('result_gte.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        ali_score.append(json.loads(line))


with open('ensemble_result_test.txt', 'w', encoding='utf-8') as out_f:
    for snowflake, ali in zip(snowflake_score, ali_score):
        id2score = defaultdict(float)
        for k, v in snowflake.items():
            id2score[k] += v
        for k, v in ali.items():
            id2score[k] += v

        id2score = Counter(id2score)
        # 选出得分最高的20个id
        top_20_pids = [each[0] for each in id2score.most_common(20)]
        out_f.write(','.join(top_20_pids) + '\n')
