import json
import numpy as np
import torch
from tqdm import tqdm

# 加载嵌入
loaded_embeddings = np.load('AQA-test-public/embeddings.npy')
loaded_query_embeddings = np.load('AQA-test-public/query_embeddings.npy')

# 加载 PID 和文章标题摘要的映射
with open('AQA-test-public/pid_to_title_abs_update_filter.json', 'r', encoding='utf-8') as f:
    pids_to_title_abstract = json.load(f)
    pids = [k for k, v in pids_to_title_abstract.items()]

# 将嵌入转换为PyTorch张量，并移动到GPU
loaded_embeddings = torch.tensor(loaded_embeddings).cuda()
loaded_query_embeddings = torch.tensor(loaded_query_embeddings).cuda()

# 处理每个 query 并找到 top 20 的相似度
with open('AQA-test-public/result_test.txt', 'w', encoding='utf-8') as out_f:
    for query_idx in tqdm(range(len(loaded_query_embeddings))):
        # 使用预加载的查询嵌入
        query_embedding = loaded_query_embeddings[query_idx].unsqueeze(0)  # 添加批次维度

        # 计算余弦相似度
        similarities = torch.nn.functional.cosine_similarity(query_embedding, loaded_embeddings, dim=1)

        # 找到相似度最高的 top 20 的 index
        top_20_indices = torch.topk(similarities, 20, largest=True).indices

        # 根据 index 获取对应的 pids
        top_20_pids = [pids[idx.item()] for idx in top_20_indices]

        # 将结果写入文件
        out_f.write(','.join(top_20_pids) + '\n')