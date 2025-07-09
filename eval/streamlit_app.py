import streamlit as st
import json
import re
from pathlib import Path

# 🔧 사용자 정의: JSON 폴더 경로
DATA_PATH = Path("/home/cvlab/Desktop/AgentAI/eval/results")

# 🔁 모델 키 → 이름 매핑
MODEL_MAP = {
    "llm_a": "claude_sonnet4",
    "llm_b": "chatgpt4o",
    "llm_c": "gemini_pro"
}


import streamlit as st

# 이 줄을 맨 위에 넣으세요!
st.set_page_config(layout="wide", page_title="AgentAI Viewer", page_icon="🧠")

# 📦 모든 JSON 파일 로드
json_files = sorted(DATA_PATH.glob("*.json"))
query_map = {f.stem: f for f in json_files}

# 🧭 사이드바에서 Query 선택
st.sidebar.title("🔍 Query Navigation")
selected_query = st.sidebar.selectbox("Select Query", list(query_map.keys()))


# 📋 평가 기준 추가
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
with open(query_map[selected_query], 'r') as f:
    data = json.load(f)

query_text = data.get("query_text", "")
responses = data.get("responses", {})
votes = data.get("votes", {})
majority = data.get("majority_vote", "")

# 🎯 질문 추출
ask_match = re.search(r"A user asks:\n[\"“](.+?)[\"”]", query_text, re.DOTALL)
user_ask = ask_match.group(1).strip() if ask_match else "(사용자 질문을 찾을 수 없습니다.)"

# 🎯 모델 목록 추출
model_block_match = re.search(r"### Recommended AI Models:\s*\n(.+)", query_text, re.DOTALL)
models_raw = model_block_match.group(1).strip() if model_block_match else "(모델 목록 없음)"
models_clean = re.findall(r"- \*\*(.*?)\*\*\n\s*Paper: (.*)", models_raw)
models_md = "\n".join([f"- **{name}**\n  Paper: {link}" for name, link in models_clean]) if models_clean else models_raw

# 📝 사용자 질문 표시
st.markdown("## 🙋 사용자 질문")
st.info(f"**\"{user_ask}\"**")

# 🤖 모델 목록 표시
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