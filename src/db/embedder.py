from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/e5-large-v2')

def embed(text):
    return model.encode(text, normalize_embeddings=True)