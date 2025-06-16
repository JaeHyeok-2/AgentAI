from transformers import AutoTokenizer, AutoModel
import torch
from typing import List

device = "cuda"
model_id = "intfloat/e5-mistral-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

def embed(texts: List[str]):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]  # [CLS] embedding
    return torch.nn.functional.normalize(embeddings, dim=1).cpu().numpy()