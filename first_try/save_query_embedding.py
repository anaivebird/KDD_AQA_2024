from sentence_transformers import SentenceTransformer
import json
import numpy as np

queries = []
with open('AQA/qa_valid_wo_ans.txt', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        query = data['question'] + ' ' + data['body']
        queries.append(query)

model = SentenceTransformer("embedding_output2").cuda()

embeddings = model.encode(queries, show_progress_bar=True)

np.save('AQA/query_embeddings.npy', embeddings)

# To load the embeddings back from the file, you can use:
# loaded_query_embeddings = np.load('AQA/query_embeddings.npy')