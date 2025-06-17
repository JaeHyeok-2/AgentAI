# src/db/vectordb.py
"""
통합 VectorDB
─────────────
index_dict = {
    "models": { "index": faiss.Index, "docs": list, "boost": -0.05 },
    "arxiv" : { "index": faiss.Index, "docs": list, "boost":  0.0  }
}
search() 한 번 호출로 두 인덱스를 모두 검색 → 거리 + boost
"""

import faiss, json, numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # src/
data_dir = BASE / "data"

def load_index_and_docs(index_name, json_name):
    idx = faiss.read_index(str(data_dir / index_name))
    docs = json.load(open(data_dir / json_name, encoding="utf-8"))
    return idx, docs

# ── 인덱스 불러오기 ─────────────────
index_dict = {
    "models": {
        "index": load_index_and_docs("model_qa_index.faiss",
                                     "model_qa_data.json")[0],
        "docs" : load_index_and_docs("model_qa_index.faiss",
                                     "model_qa_data.json")[1],
        "boost": -0.1        # 우선 순위 ↑
    },
    "arxiv": {
        "index": load_index_and_docs("model_arxiv_100_index.faiss",
                                     "model_arxiv_100_data.json")[0],
        "docs" : load_index_and_docs("model_arxiv_100_index.faiss",
                                     "model_arxiv_100_data.json")[1],
        "boost": 0.0          # 그대로
    }
}

# ── 검색 함수 ───────────────────────
def search(query_vec: np.ndarray, k_each: int = 5, k_final: int = 5):
    """
    • 두 인덱스를 모두 검색하고 boost 반영 후 상위 k_final 문서 반환
    • 연도 필터 없음 (데이터셋이 이미 2023–2025)
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