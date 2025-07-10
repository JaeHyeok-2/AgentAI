# src/rag/retriever.py
import json
import os 
import sys
import numpy as np
sys.path.append("..")

from prompt import build_prompt
from db.embedder import embed
from db.vectordb import search_split
from sentence_transformers import CrossEncoder

# Cross-Encoder 로드
reranker = CrossEncoder("BAAI/bge-reranker-large", device="cpu")

def retrieve_models_and_papers(query: str,
                                k_models: int = 5,
                                k_arxiv: int = 5,
                                use_rerank: bool = True):
    q_vec = embed([f"query: {query}"])[0]
    models, papers = search_split(q_vec, k_models=k_models, k_arxiv=k_arxiv)

    if not use_rerank:
        return _dedup(models)[:k_models], _dedup(papers)[:k_arxiv]

    def rerank_group(group):
        pairs = [
            [query, f"{d.get('Model Unique Name')}. {d.get('Summary_update') or d.get('Summary', '')}"]
                for d in group
            ]
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

    # 쿼리 로딩
    with open("/Users/jaeyoung/kuaicv/AgentAI/dataset/query_extract/model_queries_CNAPS_159_query3.json", encoding="utf-8") as f:
        model_queries = json.load(f)
    
    
    # 결과 저장 경로
    base_output_dir = "/Users/jaeyoung/kuaicv/AgentAI/output/prompts_by_model_query"
    os.makedirs(base_output_dir, exist_ok=True)

    # 전체 처리
    for model in model_queries:
        # model_name = model.get("Model Unique Name", "Unknown_Model").replace("/", "_")
        model_name = model.get("Model Unique Name")
        model_dir = os.path.join(base_output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        query_dir = os.path.join(model_dir, "Query3")
        os.makedirs(query_dir, exist_ok=True)

        for i in range(3, 4):
            query = model.get(f"Query{i}", "").strip()
            if not query:
                print("empty query")
                continue  # 빈 쿼리 건너뛰기
            try:
                models, papers = retrieve_models_and_papers(query, k_models=3, k_arxiv=5, use_rerank=True)
                prompt = build_prompt(query, selected_docs=None, model_list=models)


                query_path = os.path.join(query_dir, f"Query{i}.txt")

                print(query_path)

                with open(query_path, "w", encoding="utf-8") as f:
                    f.write(prompt)

                print(f"Saved: {query_path}")

            except Exception as e:
                print(f"Error on {model_name} Query{i}: {e}")