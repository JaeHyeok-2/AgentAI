# src/db/builder.py

import os, json, faiss, numpy as np, sys
from tqdm import tqdm

sys.path.append("..")
from utils.file_io import load_json, save_json
from embedder import embed, MODEL_ID  # 임베딩 모델 ID

MODEL_NAME = MODEL_ID.split("/")[-1]
SAVE_DIR = os.path.join("..", "data", MODEL_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

# 💡 처리할 데이터셋 목록
DATASETS = [
    {
        "name": "arxiv",
        "input": "/home/cvlab/Desktop/AgentAI/dataset/arxiv_sample_2023_after.json",
        "index_file": os.path.join(SAVE_DIR, "arxiv_index.faiss"),
        "output_json": os.path.join(SAVE_DIR, "arxiv_data.json"),
    },
    {
        "name": "new_models",
        "input": "/home/cvlab/Desktop/AgentAI/dataset/New_AI_model_no_query.json",
        "index_file": os.path.join(SAVE_DIR, "New_AI_model_no_query.faiss"),
        "output_json": os.path.join(SAVE_DIR, "New_AI_model_no_query.json"),
    },
]

def build_vector_db_for(dataset: dict):
    data = load_json(dataset["input"])
    vecs, valid_docs = [], []
    for e in tqdm(data, desc=f"Embedding {dataset['name']}"):
        title = e.get("Model Unique Name", "").strip()
        summary = e.get("Summary", "").strip()
        if not summary:
            continue
        text = f"{title} {summary}"
        vec = embed([text])[0]
        vecs.append(vec)
        valid_docs.append(e)

    vecs = np.vstack(vecs).astype("float32")
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, dataset["index_file"])
    save_json(valid_docs, dataset["output_json"])
    print(f"✅ [{dataset['name']}] Saved FAISS → {dataset['index_file']}")
    print(f"✅ [{dataset['name']}] Saved JSON  → {dataset['output_json']}")

def main():
    for ds in DATASETS:
        build_vector_db_for(ds)

if __name__ == "__main__":
    main()