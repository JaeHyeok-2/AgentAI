# src/db/vectordb.py

import faiss
import json
import numpy as np
from pathlib import Path
from db.embedder import MODEL_ID

# 🔧 임베딩 모델 디렉토리
MODEL_NAME = MODEL_ID.split("/")[-1]
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data" / "158_model" / MODEL_NAME

def load_index_and_docs(index_path, json_path):
    index = faiss.read_index(str(index_path))
    docs = json.load(open(json_path, encoding="utf-8"))
    return index, docs

# ── 인덱스 분리 불러오기 ─────────────────────────────
index_dict = {
    "models": {
        "index": load_index_and_docs(
            DATA_DIR / "merged_data.faiss",
            DATA_DIR / "merged_data.json"
        )[0],
        "docs": load_index_and_docs(
            DATA_DIR / "merged_data.faiss",
            DATA_DIR / "merged_data.json"
        )[1],
    },
    "arxiv": {
        "index": load_index_and_docs(
            DATA_DIR / "arxiv_index.faiss",
            DATA_DIR / "arxiv_data.json"
        )[0],
        "docs": load_index_and_docs(
            DATA_DIR / "arxiv_index.faiss",
            DATA_DIR / "arxiv_data.json"
        )[1],
    }
}

# ── 분리 검색 함수 추가 ─────────────────────────────
def search_split(query_vec: np.ndarray, k_models: int = 5, k_arxiv: int = 5):
    """
    FAISS 인덱스를 모델/논문 각각 분리하여 검색.
    반환: (models_docs[], arxiv_docs[])
    """
    results = {}

    for key, k in [("models", k_models), ("arxiv", k_arxiv)]:
        # key = "models", "arxiv"
        # k = 5, 5
        # 총 2번 loop 한다.
        cfg = index_dict[key]
        D, I = cfg["index"].search(np.array([query_vec], dtype="float32"), k)
        results[key] = [cfg["docs"][i] for i in I[0]]

    return results["models"], results["arxiv"]