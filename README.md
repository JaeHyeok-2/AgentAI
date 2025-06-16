# 🧠 AgentAI - Reflective RAG System

AgentAI is a research-aware RAG (Retrieval-Augmented Generation) pipeline designed to answer natural language questions using academic papers (2023–2025 arXiv + curated model summaries). It includes self-reflection and feedback to improve LLM outputs, inspired by AI Scientist-style workflows.

---

## 📁 Directory Structure
AgentAI/
└── src/
├── main.py                 # Entry point to run RAG + feedback loop
├── config.py               # (optional) Configs
├── data/
│   ├── model_qa.json       # Curated models with queries + summary
│   ├── arxiv_db.json       # 2023–2025 arXiv papers with summaries
│   └── index.faiss         # FAISS index built on summaries
├── db/
│   ├── embedder.py         # SentenceTransformer-based embedder
│   ├── vectordb.py         # Search interface over FAISS
│   └── builder.py          # Builds FAISS index from JSON
├── rag/
│   ├── retriever.py        # Retrieves relevant documents
│   ├── prompt.py           # Builds RAG + context prompt
│   └── answer.py           # Generates and reflects LLM responses
└── utils/
├── file_io.py          # Load/save helpers
└── logger.py           # (optional) for logs

---


## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r src/requirements.txt

2. Prepare data
	•	Make sure model_qa.json and arxiv_db.json are in src/data/
	•	To (re)build FAISS index:
    
    python src/db/builder.py

3. Run inference
python src/main.py


💡 Features
	•	🔍 Vector search over paper summaries (2023–2025 arXiv + curated models)
	•	🧠 GPT-4o-based RAG answering
	•	🪞 LLM self-critique + improved second-pass output
	•	📚 Easily extendable for more domains (e.g., patents, web docs, reports)

⸻
