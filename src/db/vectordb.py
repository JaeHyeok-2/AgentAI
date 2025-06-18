# src/db/vectordb.py

import faiss, json, numpy as np
from pathlib import Path
from db.embedder import MODEL_ID  # 현재 사용 중인 임베딩 모델 ID

# 🔧 임베딩 모델명에 따라 해당 하위 디렉토리에서 불러오기
MODEL_NAME = MODEL_ID.split("/")[-1]
BASE = Path(__file__).resolve().parent.parent  # src/
DATA_DIR = BASE / "data" / MODEL_NAME         # e.g., data/e5-large-v2/

def load_index_and_docs(index_path, json_path):
    index = faiss.read_index(str(index_path))
    docs  = json.load(open(json_path, encoding="utf-8"))
    return index, docs

# ── 인덱스 불러오기 ─────────────────────
index_dict = {
    "models": {
        "index": load_index_and_docs(
            DATA_DIR / "New_AI_model_no_query.faiss",
            DATA_DIR / "New_AI_model_no_query.json"
        )[0],
        "docs": load_index_and_docs(
            DATA_DIR / "New_AI_model_no_query.faiss",
            DATA_DIR / "New_AI_model_no_query.json"
        )[1],
        "boost": 0.0
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
        "boost": 0.0
    }
}

# ── 검색 함수 ──────────────────────────
def search(query_vec: np.ndarray, k_each: int = 5, k_final: int = 5):
    """
    두 인덱스(models + arxiv)를 모두 검색하고,
    거리 + boost 기준으로 상위 k_final 문서 반환
    """
    hits = []

    for cfg in index_dict.values():
        D, I = cfg["index"].search(
            np.array([query_vec], dtype="float32"), k_each
        )
        for dist, idx in zip(D[0], I[0]):
            doc = cfg["docs"][idx]
            hits.append((dist + cfg["boost"], doc))

    hits.sort(key=lambda x: x[0])
    return [doc for _, doc in hits[:k_final]]