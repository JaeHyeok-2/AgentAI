import streamlit as st
import json
import re
from pathlib import Path
import glob
import os
from collections import defaultdict

# 📂 경로 설정
DATA_PATH = Path("/Users/jaeyoung/kuaicv/AgentAI/output/prompts_by_model_query")
pattern = os.path.join(DATA_PATH, "**", "*.json")
json_paths = sorted(glob.glob(pattern, recursive=True))

# ✅ 모델별 query 구조화
query_map = defaultdict(dict)
for f in json_paths:
    path = Path(f)
    model_name = path.parts[-3]
    query_name = path.parts[-2]
    query_map[model_name][query_name] = path

# 🔁 LLM 이름 매핑
MODEL_MAP = {
    "llm_a": "claude_sonnet4",
    "llm_b": "chatgpt4o",
    "llm_c": "gemini_pro"
}

# ✅ 사용자 질문 추출 함수
def extract_user_question(query_text):
    match = re.search(r'["“](.+?)["”]', query_text, re.DOTALL)
    return match.group(1).strip() if match else "(사용자 질문을 찾을 수 없습니다.)"

# 🧭 UI 시작
st.set_page_config(layout="wide", page_title="AgentAI Viewer", page_icon="🧠")
st.sidebar.title("🔍 Query Navigation")

# 🔽 모델 선택
selected_model = st.sidebar.selectbox("모델 선택", list(query_map.keys()))

# 🔽 쿼리 선택
if selected_model:
    query_names = list(query_map[selected_model].keys())
    selected_query = st.sidebar.selectbox("쿼리 선택", query_names)

    # 📋 평가 기준
    with st.sidebar.expander("📋 평가 기준 보기"):
        st.markdown("""
**🧪 평가 기준 (각 항목당 10점 만점)**

| 번호 | 항목 | 설명 |
|------|------|------|
| 1 | 명확성 및 가독성 | 설명이 명확하고 구조적으로 잘 정리되었는가? |
| 2 | 정확성 및 완전성 | 요구된 모든 항목이 빠짐없이 포함되었는가? |
| 3 | CNAPS 스타일 워크플로우 | 분기(branch), 병합(merge) 등 시냅스 구조가 반영되었는가? |
| 4 | 제공 모델만 사용 | 문제에서 제시한 모델만을 사용했는가? |
| 5 | 해석 가능성과 설득력 | 선택 모델의 근거와 설명이 설득력 있었는가? |
""")

    # 📄 JSON 로드
    json_path = query_map[selected_model][selected_query]
    with open(json_path, 'r') as f:
        data = json.load(f)

    query_text = data.get("query_text", "")
    responses = data.get("responses", {})
    votes = data.get("votes", {})
    majority = data.get("majority_vote", "")

    # 🎯 사용자 질문 추출
    user_ask = extract_user_question(query_text)

    # 🎯 추천 모델 목록 추출
    model_block_match = re.search(r"### Recommended AI Models:\s*\n(.+)", query_text, re.DOTALL)
    models_raw = model_block_match.group(1).strip() if model_block_match else "(모델 목록 없음)"
    models_clean = re.findall(r"- \*\*(.*?)\*\*\n\s*Paper: (.*)", models_raw)
    models_md = "\n".join([f"- **{name}**\n  Paper: {link}" for name, link in models_clean]) if models_clean else models_raw

    # 🔎 사용자 질문 출력
    st.markdown("## 🙋 사용자 질문")
    st.info(f"**\"{user_ask}\"**")

    # 📚 추천된 모델 출력
    st.markdown("## 🧠 추천된 AI 모델 목록")
    st.code(models_md, language="markdown")

    # 📊 모델 응답 비교
    st.markdown("## 🤖 Model Responses")
    for raw_key in ["llm_a", "llm_b", "llm_c"]:
        response = responses.get(raw_key, "(No response found)")
        mapped_name = MODEL_MAP.get(raw_key, raw_key)
        voted_by = [model for model, v in votes.items() if v == raw_key]
        majority_flag = "🌟 **Majority Vote**" if raw_key == majority else ""

        with st.expander(f"🧠 {mapped_name}", expanded=True):
            st.markdown(response, unsafe_allow_html=True)
            st.markdown(f"✅ **Voted by**: {', '.join(voted_by) if voted_by else 'None'}")
            if majority_flag:
                st.markdown(majority_flag)
