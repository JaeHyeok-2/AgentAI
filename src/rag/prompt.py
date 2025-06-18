from typing import List, Dict

def build_prompt(query: str, selected_docs: List[Dict]) -> str:
    context = "\n".join(
        f"Model: {doc.get('Model Unique Name', 'N/A')}\n"
        f"Paper: {doc.get('Paper', 'N/A')}\n"
        f"GitHub: {doc['GitHub']}\n" if doc.get("GitHub") else
        f"Model: {doc.get('Model Unique Name', 'N/A')}\n"
        f"Paper: {doc.get('Paper', 'N/A')}\n"
        f"" +
        (f"Summary: {doc['Summary']}" if doc.get('Summary') else '')
        for doc in selected_docs
    )

    return f"""You are an AI scientist.

A user has asked the following question:
"{query}"

Recommend a suitable AI model for this task, and describe:
1. What task the user is performing.
2. How this model works in CNAPS AI-like workflows (input → model → result).
3. List relevant papers and tools they can use, including any available GitHub links.

Use only the following selected models and papers for reference:
{context}

Answer:"""