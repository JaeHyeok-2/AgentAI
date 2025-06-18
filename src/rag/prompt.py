from typing import List, Dict

def build_prompt(query: str, selected_docs: List[Dict]) -> str:
    context = "\n".join(
        f"Title: {doc['Model Unique Name']}\n"
        f"Summary: {doc['Summary']}"
        for doc in selected_docs
    )
    return f"""
You are an AI scientist. Use only the following selected papers to answer the user's question.

Context:
{context}

Question:
{query}

Answer:
"""