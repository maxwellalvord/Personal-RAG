# MD files have been given an initial clean on noisey html tags, full cleaning is defferred until testing is complete and impact can be measured on need to clean data.
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = []
for filename in os.listdir("data"):
    if filename.endswith(".md"):
        text = open(f"data/{filename}").read()
        chunks.append({"source": filename, "text": text})

texts = [c["text"] for c in chunks]
embeddings = model.encode(texts)
for chunk, vector in zip(chunks, embeddings):
    chunk["embedding"] = vector             # Attach each vector to the chunk for later retrieval.



print(len(chunks[0]["embedding"]))
print(chunks[0]["embedding"].shape)