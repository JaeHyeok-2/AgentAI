import streamlit as st
import json
import time
import os
from retriever import retrieve_models_and_papers
from prompt import build_prompt
from answer import generate_answer_with_feedback

st.set_page_config(page_title="RAG 쿼리 실행기", layout="centered")
st.title("📚 모델 1개 + 논문 추천 + GPT 응답 (선택 실행)")

# 🔹 쿼리 불러오기
with open("/home/cvlab/Desktop/AgentAI/dataset/model_queries_only.json", encoding="utf-8") as f:
    model_queries = json.load(f)

query_pool = []
for model in model_queries:
    model_name = model.get("Model Unique Name", "Unknown Model")
    for i in range(1, 4):
        q = model.get(f"Query{i}", "").strip()
        if q:
            query_pool.append((q, model_name))

if "use_rerank" not in st.session_state:
    st.session_state["use_rerank"] = True
use_rerank = st.checkbox("🚀 Cross-Encoder Rerank 사용", value=st.session_state["use_rerank"])

if "results" not in st.session_state:
    st.session_state["results"] = []

if "cache" not in st.session_state:
    st.session_state["cache"] = {}

st.markdown(f"총 {len(query_pool)}개의 쿼리에 대해 실행 가능합니다.")

# 🔹 쿼리 순차 실행
for i, (query, model_name) in enumerate(query_pool, 1):
    query_key = f"query_{i}"
    st.markdown(f"---\n### 🔍 Query {i}: {query}")
    st.markdown(f"📌 원래 모델: `{model_name}`")

    if st.button(f"🔍 문서 추천 실행 (Query {i})"):
        with st.spinner("⏳ 문서 및 모델 추천 중..."):
            try:
                models, papers = retrieve_models_and_papers(query, k_models=1, k_arxiv=5, use_rerank=use_rerank)
                prompt = build_prompt(query, papers[:3])
                st.session_state["cache"][query_key] = {
                    "query": query,
                    "model_name": model_name,
                    "model": models[0] if models else None,
                    "papers": papers,
                    "prompt": prompt
                }
            except Exception as e:
                st.error(f"❌ 오류: {e}")
                continue

    if query_key in st.session_state["cache"]:
        cached = st.session_state["cache"][query_key]
        selected_model = cached["model"]
        papers = cached["papers"]
        prompt = cached["prompt"]

        st.markdown("### 🛠 추천 AI 모델")
        if selected_model:
            url = selected_model.get("GitHub") or selected_model.get("HuggingFace") or "#"
            st.markdown(f"- **[{selected_model['Model Unique Name']}]({url})**")
        else:
            st.markdown("_추천 모델 없음_")

        st.markdown("### 📖 관련 논문 Top 5")
        for idx, p in enumerate(papers):
            dist = p.get("dist", 0.0)
            score = p.get("score")
            score_str = f", Score={score:.3f}" if score is not None else ""
            st.markdown(f"**{idx+1}. [{p['Model Unique Name']}]({p['Paper']})** (L2={dist:.3f}{score_str})")

        st.markdown("### 📝 생성된 프롬프트")
        st.code(prompt, language="text")

        # GPT 실행 버튼
        if st.button(f"🤖 GPT 응답 생성 (Query {i})"):
            with st.spinner("LLM 응답 생성 중..."):
                try:
                    answer = generate_answer_with_feedback(prompt)
                    st.success("✅ GPT 응답 생성 완료")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"LLM 오류: {e}")
                    answer = ""

            st.session_state["results"].append({
                "query": query,
                "original_model": model_name,
                "recommended_model": selected_model.get("Model Unique Name") if selected_model else None,
                "recommended_papers": [p["Model Unique Name"] for p in papers],
                "prompt": prompt,
                "answer": answer
            })

# 🔹 결과 저장
if st.session_state["results"]:
    if st.button("💾 전체 결과 JSON 저장"):
        save_path = "/home/cvlab/Desktop/AgentAI/output/llm_rag_results.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(st.session_state["results"], f, indent=2, ensure_ascii=False)
        st.success(f"📁 저장 완료 → {save_path}")