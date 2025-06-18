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

# 🔁 함수에 use_rerank 옵션 추가
def retrieve_relevant_docs(query: str,
                           k_models: int = 15,
                           k_arxiv:  int = 15,
                           k_final:  int = 5,
                           use_rerank: bool = True):
    """
    use_rerank=False인 경우 Cross-Encoder 없이 FAISS 거리 기준으로 반환
    """
    q_vec = embed([f"query: {query}"])[0]        # (D,)
    rough_docs = _vec_search(q_vec,
                             k_models=k_models,
                             k_arxiv=k_arxiv)

    print("\n🔍 후보 문서:")
    for d in rough_docs[:min(10, len(rough_docs))]:
        print("→", d["Model Unique Name"])

    if not use_rerank:
        return rough_docs[:k_final]  # Cross-Encoder 사용 안 함

    print("\n🚀 Cross-Encoder reranking 시작...")
    pairs = [[query, f"{d['Model Unique Name']} . {d['Summary']}"]
             for d in rough_docs]
    scores = reranker.predict(pairs, batch_size=8)

    reranked = [
        d for _, d in sorted(
            zip(scores, rough_docs),
            key=lambda x: x[0],
            reverse=True
        )
    ][:k_final]
    return reranked

if __name__ == "__main__":
    q = ("I watched just a few videos on YouTube, and suddenly it’s recommending ones I actually love. How does it know?")

    print("\n📌 [1] Rerank OFF (순수 FAISS)")
    docs_faiss = retrieve_relevant_docs(q, k_final=5, use_rerank=False)
    for d in docs_faiss:
        print("FAISS →", d["Model Unique Name"])

    print("\n📌 [2] Rerank ON (Cross-Encoder)")
    docs_rerank = retrieve_relevant_docs(q, k_final=5, use_rerank=True)
    for d in docs_rerank:
        print("Rerank →", d["Model Unique Name"])