# src/rag/prompt.py
from typing import List, Dict

def build_prompt(query: str, selected_docs: List[Dict], model_list: List[Dict]) -> str:
    context = ""

    # ▶ 추천된 모델 정보
    if model_list:
        context += "### Recommended AI Models:\n"
        for model in model_list:
            context += f"- **{model.get('Model Unique Name','N/A')}**\n"
            context += f"  Paper: {model.get('Paper','N/A')}\n"
            if model.get("GitHub"):
                context += f"  GitHub: {model['GitHub']}\n"
            context += "\n"

    # ▶ 논문 정보
    if selected_docs:
        context += "### Related Papers:\n"
        for doc in selected_docs:
            context += f"- **{doc.get('Model Unique Name','N/A')}**\n"
            context += f"  Paper: {doc.get('Paper','N/A')}\n"
            if doc.get("GitHub"):
                context += f"  GitHub: {doc['GitHub']}\n"
            if doc.get("Summary"):
                context += f"  Summary: {doc['Summary']}\n"
            context += "\n"

    return f"""You are an expert AI scientist and architect of a CNAPS‑style multi‑module workflow.
Here, CNAPS means a synapse‑like network of AI models connected in series or parallel—but without any built-in decision logic or condition-based branching; it simply executes the pre-defined model chain like block coding.
Only the models you explicitly connect are run. Unconnected models are never used.

A user asks:
"{query}"

Using ONLY the provided models and papers below, respond in full detail.
You do not need to use all listed models—only those essential to the user's goal.

---
## 1. Core Task
Summarize the user's intent in one or two sentences.
List any sub-goals (e.g., deblurring, colorization, style transfer).

---
## 2. CNAPS-style Workflow
Design a high-level CNAPS workflow that:
- Connects only the minimum necessary model(s) required for the task.
- Does not include unused branches or models.
- Maintains CNAPS structure (series or simple parallel flows) but no conditional routing logic.

Your answer must include:

### A. High-Level Overview
Explain in natural language how the chain works: what the input is, which model(s) process it, and what the output is.

### B. Visual Flow Diagram (text-based blocks)
Use a simple flow chart style:
[Input: ...]
   |
   v
[Model A (purpose)]
   |
   v
[Model B (purpose)]         # Optional second stage if needed
   |
   v
[Final Output (goal + confidence)]
Only include the model(s) you selected—no extras.

### C. Justify Your Design
Explain why each selected model is necessary and why others were omitted, with references to their papers or GitHub/ArXiv links.

# \n{context}\nAnswer:
# """

