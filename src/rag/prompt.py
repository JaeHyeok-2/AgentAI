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

    return f"""You are an expert AI scientist and architect of a CNAPS‑style single-model workflow.  
Here, CNAPS means a synapse‑like network of AI models connected in series or parallel—but without any built-in decision logic or condition-based branching; it simply executes the pre-defined model chain like block coding.

⚠️ Important: You must select and use only ONE model from the provided list.  
Do not chain multiple models. Unconnected models are never used.  
Among the provided models, choose only the **single most appropriate model** to fulfill the user's goal.

A user asks:  
"{query}"

Using ONLY the provided models and papers below, respond in full detail.  
Ignore any model that is not strictly necessary.  
This workflow should be achievable using exactly one model.

---
## 1. Core Task  
Summarize the user's intent in one or two sentences.  
List any sub-goals (e.g., deblurring, colorization, style transfer).

---
## 2. CNAPS-style Workflow (Single-Model)  
Design a high-level CNAPS workflow that:  
- Uses exactly one model to fulfill the task.  
- Does not include unused models.  
- Maintains CNAPS structure (simple, linear execution).

Your answer must include:

### A. High-Level Overview  
Explain in natural language how the input is processed by the selected model and what the final output is.

### B. Visual Flow Diagram (text-based blocks)  
Use the format below to show the model chain (with only one model):

[Input: ...]  
   |  
   v  
[Model A (purpose)]  
   |  
   v  
[Final Output (goal + confidence)]

### C. Justify Your Design  
Explain clearly why this one model is sufficient for the user's task, and why no other models were used.  
Cite the model’s paper or GitHub/ArXiv reference.

# \n{context}\nAnswer:
# """

