import json
import faiss
import numpy as np
from db.embedder import embed

with open("src/data/arxiv_db.json") as f:
    docs = json.load(f)

embeddings = [embed(doc["summary"]) for doc in docs]
embeddings = np.vstack(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "src/data/index.faiss")
print("✅ FAISS index 생성 완료")