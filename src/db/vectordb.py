import faiss
import numpy as np
import json

with open("src/data/arxiv_db.json") as f:
    docs = json.load(f)

index = faiss.read_index("src/data/index.faiss")

def search(query_embedding, top_k=5):
    D, I = index.search(np.array([query_embedding]), top_k)
    return [docs[i] for i in I[0]]