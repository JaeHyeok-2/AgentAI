# src/db/builder.py

import json
import faiss
import numpy as np
from tqdm import tqdm
from embedder import embed
import sys
sys.path.append("..")
from utils.file_io import load_json, save_json

DATA_PATH = "/home/cvlab/Desktop/AgentAI/dataset/New_AI_Model.json"
INDEX_PATH = "../data/model_qa_index.faiss"
CLEANED_PATH = "../data/model_qa_data.json"

def build_vector_db():
    data = load_json(DATA_PATH)

    embeddings = []
    valid_docs = []

    for entry in tqdm(data):
        summary = entry.get("Summary", "").strip()
        if summary:
            vec = embed(summary)
            embeddings.append(vec)
            valid_docs.append(entry)

    embeddings = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    save_json(valid_docs, CLEANED_PATH)
    print(f"✅ Indexed {len(valid_docs)} documents → saved to {INDEX_PATH}")

if __name__ == "__main__":
    build_vector_db()