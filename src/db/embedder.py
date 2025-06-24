from transformers import AutoTokenizer, AutoModel
import torch
from typing import List

device = "cuda"
MODEL_ID = "jinaai/jina-embeddings-v3"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model     = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True).to(device)

def embed(texts: List[str]):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(emb, dim=1).cpu().numpy()