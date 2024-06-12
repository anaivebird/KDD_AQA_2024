import torch
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("embedding_output2", trust_remote_code=True).cuda()

# Calculate the total number of parameters
total_params = sum(p.numel() for p in model.parameters())

print(f"Total number of parameters: {total_params}")