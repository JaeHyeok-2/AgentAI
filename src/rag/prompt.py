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

    return f"""
You are an expert AI scientist and architect of a CNAPS‑style multi‑module workflow.  
Here, CNAPS means a **synapse‑like branching network** of AI models working together—not a simple linear pipeline.

A user asks:
"{query}"

**Using ONLY the provided models and papers in the context below, answer the following in full detail.**  
You do **not** need to use all listed models—only those that are clearly relevant to the user's goal.

---

## 1. Core Task  
Summarize what the user wants to achieve in one or two sentences.  
List any sub-goals involved (e.g., structure preservation, texture realism, style matching).

---

## 2. CNAPS-style Workflow  
Design a **high-level synaptic workflow** that clearly shows branching, merging, or conditional paths.  
Your answer **must include all 3 parts below**:

---

### A. High-Level Overview  
Explain how the system works in natural language:  
What flows in, how it branches, what gets combined, and what comes out.

---

### B. Visual Flow Diagram (block-style, text-based)  
Use a simple visual flow chart using boxed steps, arrows, and indentation.  
Use this style:
# Here, CNAPS means a **synapse‑like branching network** of AI models working together—not a simple linear pipeline.

# A user asks:
# "{query}"

# **Using ONLY the provided models and papers in the context below, do the following:**

# 1. **Identify the core task or goal** implied by the user’s request.  
# 2. **Design a CNAPS-style synaptic workflow**:
#    - Describe how input is routed to one or more modules.
#    - Explain how modules branch, interact, merge, or loop.
#    - Define each module’s intermediate and final output formats/include examples.
# 3. **Justify your design** with references to the papers and tools (include GitHub or ArXiv links listed).

# \n{context}\nAnswer:
# """

