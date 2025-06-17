# src/db/builder.py
import json, faiss, numpy as np, sys
from tqdm import tqdm
sys.path.append("..")
from utils.file_io import load_json, save_json
from embedder import embed

DATA_PATH   = "/home/cvlab/Desktop/AgentAI/dataset/arxiv_sample_100.json"
INDEX_PATH  = "../data/model_arxiv_100_index.faiss"
CLEANED_PATH = "../data/model_arxiv_100_data.json"

def build_vector_db():
    data = load_json(DATA_PATH)

    vecs, valid_docs = [], []

    for e in tqdm(data):
        title   = e.get("Model Unique Name", "").strip()
        summary = e.get("Summary", "").strip()

        if not summary:      # 요약 없으면 건너뜀
            continue

        text = f"{title} {summary}"   # ← 제목 + 요약
        vec  = embed([text])[0]       # embed는 리스트 입력 → (1,D) 반환

        vecs.append(vec)
        valid_docs.append(e)

    vecs = np.vstack(vecs).astype("float32")
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, INDEX_PATH)
    save_json(valid_docs, CLEANED_PATH)
    print(f"✅ Indexed {len(valid_docs)} docs → {INDEX_PATH}")

if __name__ == "__main__":
    build_vector_db()