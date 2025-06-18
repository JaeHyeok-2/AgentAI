# src/rag/retriever.py

import sys
import numpy as np
sys.path.append("..")

from db.embedder import embed
from db.vectordb import search_split
from sentence_transformers import CrossEncoder

# 🔁 Cross-Encoder 로드
reranker = CrossEncoder("BAAI/bge-reranker-large", device="cuda")

def retrieve_models_and_papers(query: str,
                                k_models: int = 5,
                                k_arxiv: int = 5,
                                use_rerank: bool = True):
    q_vec = embed([f"query: {query}"])[0]
    models, papers = search_split(q_vec, k_models=k_models, k_arxiv=k_arxiv)

    if not use_rerank:
        return _dedup(models)[:k_models], _dedup(papers)[:k_arxiv]

    def rerank_group(group):
        pairs = [[query, f"{d['Model Unique Name']}. {d['Summary'][:512]}"] for d in group]
        scores = reranker.predict(pairs, batch_size=8)
        scored = [(s, d) for s, d in zip(scores, group)]

        # 유니크한 Model Unique Name 기준으로 중복 제거
        seen = set()
        dedup_sorted = []
        for score, doc in sorted(scored, key=lambda x: x[0], reverse=True):
            name = doc.get("Model Unique Name")
            if name not in seen:
                seen.add(name)
                dedup_sorted.append(doc)
        return dedup_sorted

    reranked_models = rerank_group(models)[:k_models]
    reranked_papers = rerank_group(papers)[:k_arxiv]

    return reranked_models, reranked_papers


# 보조 함수
def _dedup(docs):
    seen = set()
    result = []
    for d in docs:
        name = d.get("Model Unique Name")
        if name not in seen:
            seen.add(name)
            result.append(d)
    return result


# ── CLI 테스트 ─────────────────────────────
if __name__ == "__main__":
    q = "I never told the shopping app what I like, but it shows me exactly what I’d pick."

    print("\n📦 모델 & 논문 추천 테스트")
    models, papers = retrieve_models_and_papers(q, use_rerank=True)

    print("\n🛠 추천된 모델:")
    for m in models:
        print("-", m["Model Unique Name"])

    print("\n📖 추천된 논문:")
    for p in papers:
        print("-", p["Model Unique Name"])