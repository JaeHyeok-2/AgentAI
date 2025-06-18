import streamlit as st
import json
import time
from retriever import retrieve_relevant_docs
from prompt import build_prompt

st.set_page_config(page_title="RAG 전체 쿼리 자동 실행", layout="centered")
st.title("📚 전체 모델의 Query 자동 실행")

# 🔹 모든 모델의 쿼리 로드
with open("/home/cvlab/Desktop/AgentAI/dataset/model_queries_only.json", encoding="utf-8") as f:
    model_queries = json.load(f)

# 🔹 전체 쿼리 (query, model name) 추출
query_pool = []

for model in model_queries:
    model_name = model.get("Model Unique Name", "Unknown Model")
    for i in range(1, 4):
        q = model.get(f"Query{i}", "").strip()
        if q:
            query_pool.append((q, model_name))

# 🔹 rerank 설정
if "use_rerank" not in st.session_state:
    st.session_state["use_rerank"] = True
use_rerank = st.checkbox("🚀 Cross-Encoder Rerank 사용", value=st.session_state["use_rerank"])

st.markdown(f"### 🔍 총 {len(query_pool)}개의 쿼리에 대해 추천을 수행합니다.")

# 🔹 실행 버튼
if st.button("📥 전체 Query 실행"):
    for i, (query, model_name) in enumerate(query_pool, 1):
        st.markdown(f"## 🔍 Query {i}: {query}")
        st.markdown(f"📌 관련된 모델: `{model_name}`")

        with st.spinner("⏳ 검색 중..."):
            start_time = time.time()
            try:
                docs = retrieve_relevant_docs(
                    query,
                    k_final=10,
                    use_rerank=use_rerank
                )
                elapsed = time.time() - start_time
                st.success(f"✅ 완료 (⏱ {elapsed:.2f}초)")
            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")
                continue

        if not docs:
            st.warning("추천 결과가 없습니다.")
            continue

        st.markdown("### 🏆 추천 논문 Top 10")
        for idx, d in enumerate(docs):
            dist = d.get("dist", 0.0)
            score = d.get("score", None)
            score_str = f", Score={score:.3f}" if score is not None else ""
            st.markdown(f"**{idx+1}. {d['Model Unique Name']}** (L2={dist:.3f}{score_str})")

        # 상위 3개 자동 선택
        top3 = docs[:3]
        st.markdown("### ✅ 자동 선택된 상위 3개 논문")
        for d in top3:
            st.markdown(f"- **[{d['Model Unique Name']}]({d['Paper']})**")

        prompt = build_prompt(query, top3)
        st.markdown("### 📝 생성된 LLM용 프롬프트")
        st.code(prompt, language="text")

        st.markdown("---")