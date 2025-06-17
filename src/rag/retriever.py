# src/rag/retriever.py
import sys, numpy as np, faiss, json
sys.path.append("..")

from db.embedder import embed
from db.vectordb import index_dict        # 방금까지 쓰던 models / arxiv 인덱스
from sentence_transformers import CrossEncoder

# ── Cross-Encoder 로드 (GPU) ──
reranker = CrossEncoder("BAAI/bge-reranker-large", device="cuda")

def _vec_search(query_vec: np.ndarray,
                k_models: int = 30,
                k_arxiv:  int = 30,
                boost: float = -0.10):
    """두 인덱스 벡터 검색 → 거리 + boost"""
    hits = []
    # models
    Dm, Im = index_dict["models"]["index"].search(
        query_vec.reshape(1, -1).astype("float32"), k_models
    )
    for dist, idx in zip(Dm[0], Im[0]):
        hits.append((dist + boost, index_dict["models"]["docs"][idx]))

    # arXiv
    Da, Ia = index_dict["arxiv"]["index"].search(
        query_vec.reshape(1, -1).astype("float32"), k_arxiv
    )
    for dist, idx in zip(Da[0], Ia[0]):
        hits.append((dist, index_dict["arxiv"]["docs"][idx]))

    # 1차: 거리 기준 정렬
    hits.sort(key=lambda x: x[0])
    return [doc for _, doc in hits]

def retrieve_relevant_docs(query: str,
                           k_models: int = 15,
                           k_arxiv:  int = 15,
                           k_final:  int = 5):
    """
    1) 벡터 검색으로 후보 30개 확보
    2) Cross-Encoder(BGE-large)로 재정렬
    3) 상위 k_final 반환
    """
    q_vec = embed([f"query: {query}"])[0]        # (D,)
    rough_docs = _vec_search(q_vec,
                             k_models=k_models,
                             k_arxiv=k_arxiv)
    print([d['Model Unique Name'] for d in rough_docs[:10]])
    # 쌍 만들어 점수 예측
    pairs  = [[query, f"{d['Model Unique Name']} . {d['Summary']}"]
              for d in rough_docs]
    scores = reranker.predict(pairs, batch_size=8)

    # 점수 높은 순 → k_final
    reranked = [
        d for _, d in sorted(
            zip(scores, rough_docs),            # (score, doc) 튜플
            key=lambda x: x[0],                 # ← score만 기준
            reverse=True
        )
    ][:k_final]
    return reranked[:k_final]

# ── 간단 데모 ───────────────────────
if __name__ == "__main__":
    q = ("I watched just a few videos on YouTube, and suddenly it’s recommending ones I actually love. How does it know?.")
    docs = retrieve_relevant_docs(q, k_final=5)   # top-2 확인
    for d in docs:
        print(d["Model Unique Name"], "→", d["Paper"])