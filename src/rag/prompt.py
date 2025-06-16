def build_prompt(query, documents):
    context = "\n".join(f"Title: {doc['title']}\nSummary: {doc['summary']}" for doc in documents)
    return f"""
You are an AI scientist. Use the following papers to answer the user's question.

Context:
{context}

Question:
{query}

Answer:
"""