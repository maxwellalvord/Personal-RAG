import ollama
from sentence_transformers import SentenceTransformer
import numpy as np

documents = [
    "Espresso is made by forcing hot water through finely ground coffee under high pressure.",
    "Git branches let you work on a feature in isolation before merging it into the main line.",
    "Most houseplants die from overwatering, not underwatering — let the soil dry out between waterings.",
    "The James Webb Space Telescope observes the universe primarily in infrared light.",
    "Sourdough bread rises using wild yeast from a fermented starter, not commercial yeast.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(documents)

print(doc_embeddings.shape)

question = "How to change a car tire?"
q_emb = model.encode(question)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = [cosine_similarity(q_emb, doc_emb) for doc_emb in doc_embeddings]

best = int(np.argmax(scores))

THRESHOLD = 0.3

if scores[best] < THRESHOLD:
    print("No relevant information found.")
else:
    retrieved = documents[best]
    prompt = f"""Answer the question using only the context below. If the answer is not contained within the text below, say "I don't know".
    
    Context: {retrieved}

    question: {question}"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    print("\nAnswer:", response["message"]["content"])
