import streamlit as st
import json

# Sample data loading (use your actual JSON loading here)
with open("all_model_answers.json", "r", encoding="utf-8") as f:
    answers_data = json.load(f)
with open("all_queries_evaluation.json", "r", encoding="utf-8") as f:
    evaluations_data = json.load(f)

selected_model = st.sidebar.selectbox("Select a model", [m["model"] for m in answers_data])

answer_entry = next(x for x in answers_data if x["model"] == selected_model)
eval_entry = next(x for x in evaluations_data if x["model"] == selected_model)

st.title(f"📌 Model: {selected_model}")

for qnum in range(1, 4):
    query_key = f"query{qnum}"
    st.markdown(f"## 🧠 QUERY{qnum}")

    tabs = st.tabs(["LLM A", "LLM B", "LLM C"])
    for idx, llm in enumerate(["llm_A", "llm_B", "llm_C"]):
        with tabs[idx]:
            is_best = eval_entry["final_decisions"][query_key] == llm
            badge = "✅ Selected as Best" if is_best else ""
            st.markdown(f"**Model: {llm}** {badge}")
            st.text_area("Answer", value=answer_entry[query_key][llm], height=300)
            st.markdown(f"**Score:** {eval_entry['evaluations'][query_key][llm]['score']}")
            st.markdown("**Reasons:**")
            for reason in eval_entry["evaluations"][query_key][llm]["reasons"]:
                st.markdown(f"- {reason}")
