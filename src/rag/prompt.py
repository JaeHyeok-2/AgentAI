# src/rag/prompt.py
from typing import List, Dict

def build_prompt(query: str, selected_docs: List[Dict], model_list: List[Dict]) -> str:
    context = ""

    # ▶ 추천된 모델 정보 여러 개 처리
    if model_list:
        context += "### Recommended AI Models:\n"
        for model in model_list:
            context += f"Model: {model.get('Model Unique Name', 'N/A')}\n"
            context += f"Paper: {model.get('Paper', 'N/A')}\n"
            if model.get("GitHub"):
                context += f"GitHub: {model['GitHub']}\n"
            context += "\n"

    # ▶ 논문 포함
    context += "### Related Papers:\n"
    for doc in selected_docs:
        context += f"Model: {doc.get('Model Unique Name', 'N/A')}\n"
        context += f"Paper: {doc.get('Paper', 'N/A')}\n"
        if doc.get("GitHub"):
            context += f"GitHub: {doc['GitHub']}\n"
        if doc.get("Summary"):
            context += f"Summary: {doc['Summary']}\n"
        context += "\n"

    return f"""You are an AI scientist.

A user has asked the following question:
\"{query}\"

Based on the following recommended models, explain:

1. What task the user is trying to perform.
2. How the model(s) would work in a CNAPS AI-like workflow (input → model → output).
3. List relevant papers and tools (with GitHub or ArXiv links) that support your answer.

Use **only the provided models and papers**. Do not refer to outside sources.

{context}
Answer:
"""