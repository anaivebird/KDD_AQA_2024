import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from multiprocessing import Pool
import re

def remove_html_and_urls(text):
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text, flags=re.MULTILINE)
    # 去除URL
    text = re.sub(r'http\S+|www.\S+', '', text, flags=re.MULTILINE)
    return text

def encode_chunk(docs_chunk, model_name, device_id):
    torch.cuda.set_device(device_id)
    model = SentenceTransformer(model_name, trust_remote_code=True).cuda()
    embeddings_chunk = model.encode(docs_chunk, show_progress_bar=(device_id==0), batch_size=32, convert_to_tensor=True)
    return embeddings_chunk.cpu().numpy()

def main():
    with open('AQA-test-public/pid_to_title_abs_update_filter.json', 'r', encoding='utf-8') as f:
        pids_to_title_abstract = json.load(f)
        docs = [remove_html_and_urls(str(v))[:2000] for k, v in pids_to_title_abstract.items()]

    num_chunks = 8
    chunks = np.array_split(docs, num_chunks)
    model_name = "embedding_output2"

    # 使用多进程进行推理
    with Pool(num_chunks) as pool:
        results = pool.starmap(encode_chunk, [(chunk, model_name, i) for i, chunk in enumerate(chunks)])

    embeddings = np.vstack(results)
    np.save('AQA-test-public/embeddings.npy', embeddings)

if __name__ == '__main__':
    main()