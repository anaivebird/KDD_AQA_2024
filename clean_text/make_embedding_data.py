import json
import random
import math
import re

NEG_NUM_FOR_EACH_POS = 20
train_set = []

def remove_html_and_urls(text):
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text, flags=re.MULTILINE)
    # 去除URL
    text = re.sub(r'http\S+|www.\S+', '', text, flags=re.MULTILINE)
    return text

with open('AQA-test-public/pid_to_title_abs_update_filter.json', 'r', encoding='utf-8') as f:
    pids_to_title_abstract = json.load(f)

all_pids = list(set(pids_to_title_abstract.keys()))
pids_to_title_abstract = {k: remove_html_and_urls(str(v)) for k, v in pids_to_title_abstract.items()}

total_pos_num = 0
with open('AQA/qa_train.txt', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        total_pos_num += len(data['pids'])

NEG_NUM_FOR_EACH_POS = math.floor(len(all_pids) / total_pos_num)

i = 0
with open('AQA/qa_train.txt', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        for pid in data['pids']:
            neg_pids = list(set(all_pids[i * NEG_NUM_FOR_EACH_POS: (i + 1) * NEG_NUM_FOR_EACH_POS]) - set(data['pids']))
            negative_strs = [pids_to_title_abstract[pid] for pid in neg_pids]
            train_set.append({
                "query": remove_html_and_urls(data['question'] + ' ' + data['body']),
                "pos": [pids_to_title_abstract[pid]],
                "neg": negative_strs
            })
            i += 1

# Now, let's write the train_set to OUTPUT_FILE in jsonl format
with open('AQA/embedding_train_new.jsonl', 'w', encoding='utf-8') as fw:
    for item in train_set:
        item['pos'] = [text for text in item['pos']]
        item['neg'] = [text for text in item['neg']]
        fw.write(json.dumps(item, ensure_ascii=False) + '\n')  # Write each JSON record followed by a newline