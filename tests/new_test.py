# test_embedding.py

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Machine learning is a subset of artificial intelligence.",
    "Supervised learning uses labeled data."
]

embeddings = model.encode(
    texts,
    convert_to_numpy=True
)

print(embeddings.shape)