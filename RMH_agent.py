# MD files have been given an initial clean on noisey html tags, full cleaning is defferred until testing is complete and impact can be measured on need to clean data.
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import re 

model = SentenceTransformer("all-MiniLM-L6-v2")

golden_set = [
    {
        "question": "I just started working here, how do I get set up?",
        "must_contain": ["numeric ID", "Login ID"],
    },
    {
        "question": "I have some tomatoes that need to go to store 2, help me.",
        "must_contain": ["Inter-Store", "Transfer From"],
    },
    {
        "question": "I brought a bunch of grapes to the warehouse offsite but they are still showing in this stores inventory, how do I fix that?",
        "must_contain": ["Inter-Store", "Transfer From"],
    },
    {
        "question": "A shipment is coming in of some apples, help me get them in the system.",
        "must_contain": ["Supplier Order No."],
    },
    {
        "question": " A giant box of bananas we ordered is here, what do I do next?",
        "must_contain": ["Supplier Order No."],
    },
]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Corpus setup
chunks = []

def clean(text):
    text = re.sub(r"\{%.*?%\}", " ", text)  # GitBook {% hint %} templating
    text = re.sub(r"<[^>]+>", " ", text)     # every HTML tag -> a space
    text = re.sub(r"\s+", " ", text)          # collapse runs of whitespace
    return text.strip()  

for filename in os.listdir("data"):
    if filename.endswith(".md"):
        raw = open(f"data/{filename}").read()      # raw — do NOT clean yet
        pieces = raw.split("</details>")           # split while the tag still exists
        for i, piece in enumerate(pieces):
            piece = clean(piece)                    # clean each piece individually
            if piece:                               # skip empty fragments
                chunks.append({
                    "source": f"{filename}#{i}",
                    "text": piece,
                })

texts = [c["text"] for c in chunks]
t_emb = model.encode(texts)
for chunk, vector in zip(chunks, t_emb):
    chunk["embedding"] = vector             # Attach each vector to the chunk for later retrieval. 

print(len(chunks))

# worker retrieves ranked chunks for one question
def retrieve(question):
    q_emb = model.encode(question)
    for chunk in chunks:
        chunk["score"] = cosine_similarity(q_emb, chunk["embedding"])
    return sorted(chunks, key=lambda c: c["score"], reverse=True)

ranked = retrieve("I have some tomatoes that need to go to store 2, help me.")
for rank, chunk in enumerate(ranked, start=1):
    print(f"{rank}. {chunk['source']:30} {chunk['score']:.3f}")

# inspector scores the cases
def score_case(ranked, case):
    must_contain = [s.lower() for s in case["must_contain"]]
    for rank, chunk in enumerate(ranked, start=1):
        text = chunk["text"].lower()
        if all(phrase in text for phrase in must_contain):
            return 1 / rank
    return 0.0

# Manager runs the golden set and scores the results
def run_benchmark(golden_set):
    total = 0.0
    for case in golden_set:
        ranked = retrieve(case["question"])
        rr = score_case(ranked, case)
        total += rr
        top = ranked[0]
        print(f"RR={rr:.3f} | Top: {top['source']} {top['score']:.3f} | Question: {case['question']}")
    mmr = total / len(golden_set)
    print(f"\nMean Reciprocal Rank (MRR): {mmr:.3f}")

run_benchmark(golden_set)