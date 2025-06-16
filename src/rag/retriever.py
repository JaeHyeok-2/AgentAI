from db.embedder import embed
from db.vectordb import search

def retrieve_relevant_docs(query):
    query_vec = embed("query: " + query)
    return search(query_vec, top_k=5)