from sentence_transformers import SentenceTransformer
import json
import numpy as np
import re

def remove_html_and_urls(text):
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text, flags=re.MULTILINE)
    # 去除URL
    text = re.sub(r'http\S+|www.\S+', '', text, flags=re.MULTILINE)
    return text


queries = []
with open('AQA/qa_valid_wo_ans.txt', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        query = remove_html_and_urls(data['question'] + ' ' + data['body'])
        queries.append(query)

model = SentenceTransformer("embedding_output2", trust_remote_code=True).cuda()

embeddings = model.encode(queries, show_progress_bar=True, batch_size=8)

np.save('AQA/query_embeddings.npy', embeddings)

# To load the embeddings back from the file, you can use:
# loaded_query_embeddings = np.load('AQA/query_embeddings.npy')