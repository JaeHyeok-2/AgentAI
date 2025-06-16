
from rag.retriever import retrieve_relevant_docs
from rag.prompt import build_prompt
from rag.answer import generate_answer_with_feedback
import pandas as pd

if __name__ == "__main__":
    queries = [
        "I never told the shopping app what I like...",
        "I watched just a few videos on YouTube...",
        "My music app keeps picking songs I love..."
    ]

    for query in queries:
        print(f"\n🔍 Query: {query}")
        docs = retrieve_relevant_docs(query)
        prompt = build_prompt(query, docs)
        answer = generate_answer_with_feedback(prompt)
        print(f"\n🧠 Final Answer:\n{answer}")
```