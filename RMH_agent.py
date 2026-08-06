# MD files have been given an initial clean on noisey html tags, full cleaning is defferred until testing is complete and impact can be measured on need to clean data.
import os
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

question = "How do I add a new customers last name to the system?"
q_emb = model.encode(question)

chunks = []

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for filename in os.listdir("data"):
    if filename.endswith(".md"):
        text = open(f"data/{filename}").read()
        chunks.append({"source": filename, "text": text})

texts = [c["text"] for c in chunks]
t_emb = model.encode(texts)
for chunk, vector in zip(chunks, t_emb):
    chunk["embedding"] = vector             # Attach each vector to the chunk for later retrieval.

for chunk in chunks:
    chunk["score"] = cosine_similarity(q_emb, chunk["embedding"])

ranked = sorted(chunks, key=lambda c: c["score"], reverse=True)

print(ranked[0]["source"], ranked[0]["score"])
print(ranked[1]["source"], ranked[1]["score"])