from sentence_transformers import SentenceTransformer
import json
import numpy as np

with open('AQA/pid_to_title_abs_new.json', 'r', encoding='utf-8') as f:
    pids_to_title_abstract = json.load(f)
    docs = [str(v)[:500] for k, v in pids_to_title_abstract.items()]


model = SentenceTransformer("embedding_output2").cuda()

embeddings = model.encode(docs, show_progress_bar=True)

np.save('AQA/embeddings.npy', embeddings)

# To load the embeddings back from the file, you can use:
# loaded_embeddings = np.load('AQA/embeddings.npy')