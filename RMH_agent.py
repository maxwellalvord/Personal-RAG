# MD files have been given an initial clean on noisey html tags, full cleaning is defferred until testing is complete and impact can be measured on need to clean data.
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import re 
from rank_bm25 import BM25Okapi

model = SentenceTransformer("all-MiniLM-L6-v2")


golden_set = [
    # ---- add_customer_record.md (customer) ----
    {   # close
        "question": "How do I create a new customer record with their billing info?",
        "must_contain": ["Billing Information"],
    },
    {   # far
        "question": "A regular shopper wants to set up an account so we can mail their receipts. What do I do?",
        "must_contain": ["Billing Information"],
    },
    {   # far
        "question": "A shopper has two houses and wants deliveries sent to whichever they're at. How do I store both?",
        "must_contain": ["Set Primary"],
    },

    # ---- add_employee_accounts.md (user accounts) ----
    {   # close
        "question": "How do I add a new user account in RMH?",
        "must_contain": ["Login ID"],
    },
    {   # far
        "question": "I just started here and need my own way to sign in to the register. How does my manager set that up?",
        "must_contain": ["Login ID"],
    },
    {   # far
        "question": "How do I control what a new hire is allowed to do at the register versus the back office?",
        "must_contain": ["User Roles"],
    },

    # ---- add_store_supplier.md (supplier) — near-twin cluster ----
    {   # close
        "question": "How do I add a new supplier to the store?",
        "must_contain": ["Accepted Currency"],
    },
    {   # far
        "question": "We stopped buying from a vendor but I don't want to lose the history. What should I do instead of removing them?",
        "must_contain": ["Deactivating a supplier"],
    },
    {   # far
        "question": "The company we buy our produce from won't take orders under a certain dollar amount. Where do I record that?",
        "must_contain": ["Min. Order Amt."],
    },

    # ---- create_purchase_order.md (PO) — near-twin cluster ----
    {   # close
        "question": "How do I create a purchase order?",
        "must_contain": ["Supplier Order No."],
    },
    {   # far
        "question": "A shipment of apples we ordered from our vendor is arriving. How do I record that incoming order?",
        "must_contain": ["Supplier Order No."],
    },
    {   # far
        "question": "Who authorized the order — where do I note the person who requested it when buying stock?",
        "must_contain": ["Purchaser"],
    },

    # ---- create_transfer_out.md (transfer) — near-twin cluster ----
    {   # close
        "question": "How do I create a transfer out?",
        "must_contain": ["Inter-Store"],
    },
    {   # far
        "question": "I moved a bunch of grapes to our offsite warehouse but they still show in this store's stock. How do I fix that?",
        "must_contain": ["Transfer From"],
    },
    {   # far
        "question": "I need to send some tomatoes over to our other location. How do I do that in the system?",
        "must_contain": ["Inter-Store"],
    },

    # ---- create_standard_item.md (item) — near-twin cluster ----
    {   # close
        "question": "How do I add a new standard item to inventory?",
        "must_contain": ["Item Lookup Code"],
    },
    {   # far
        "question": "I want the system to warn me to restock cans when they get low. Where do I set that threshold?",
        "must_contain": ["Reorder Point"],
    },

    # ---- two extra hard cluster cases (supplier vs PO discrimination) ----
    {   # far — supplier, not PO
        "question": "Where do I save a vendor's phone number and website when I first set them up?",
        "must_contain": ["Web Page"],
    },
    {   # far — PO, not supplier
        "question": "The vendor requires a minimum spend per order and I want a warning if my order is too small. Where's that shown?",
        "must_contain": ["Min. Order Value"],
    },
]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Corpus setup

def clean(text):
    text = re.sub(r"\{%.*?%\}", " ", text)  # GitBook {% hint %} templating
    text = re.sub(r"<[^>]+>", " ", text)     # every HTML tag -> a space
    text = re.sub(r"\s+", " ", text)          # collapse runs of whitespace
    return text.strip()  

chunks = []
for filename in os.listdir("data"):
    if filename.endswith(".md"):
        text = clean(open(f"data/{filename}").read())
        chunks.append({"source": filename, "text": text})

texts = [c["text"] for c in chunks]
t_emb = model.encode(texts)
for chunk, vector in zip(chunks, t_emb):
    chunk["embedding"] = vector             # Attach each vector to the chunk for later retrieval. 

tokenized_corpus = [c["text"].lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized_corpus)

print(len(chunks))

# worker retrieves ranked chunks for one question
def retrieve(question, k=60, w_dense=1.0, w_bm25=0.0):
    q_emb = model.encode(question)
    for c in chunks:
        c["dense"] = cosine_similarity(q_emb, c["embedding"])

    bm25_scores = bm25.get_scores(question.lower().split())
    for c, s in zip(chunks, bm25_scores):
        c["bm25"] = s

    for rank, c in enumerate(sorted(chunks, key=lambda c: c["dense"], reverse=True), start=1):
        c["dense_rank"] = rank
    for rank, c in enumerate(sorted(chunks, key=lambda c: c["bm25"], reverse=True), start=1):
        c["bm25_rank"] = rank

    for c in chunks:
        c["score"] = w_dense * 1/(k + c["dense_rank"]) + w_bm25 * 1/(k + c["bm25_rank"])

    print(w_bm25, w_dense, k)
    return sorted(chunks, key=lambda c: c["score"], reverse=True)

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




# ---- CHUNKING EXPERIMENTS  ----
# Tested three strategies against the same golden set. One-chunk-per-file won.
# Docs are short and single-topic (one RMH workflow each), so they're already
# atomic — splitting them fragments a coherent idea instead of separating ideas.
#
#   Strategy                     MRR
#   one chunk per file           0.517   <- best, currently in use
#   fixed-size, swept (best 1100) <0.4
#   split on </details>          0.293   <- worst; one doc had 9 tables -> 9
#                                            shards that flooded the rankings
#
# --- Experiment A: split on </details> tag boundary ---
# for filename in os.listdir("data"):
#     if filename.endswith(".md"):
#         raw = open(f"data/{filename}").read()      # split raw: clean() strips the tag
#         pieces = raw.split("</details>")
#         for i, piece in enumerate(pieces):
#             piece = clean(piece)
#             if piece:
#                 chunks.append({"source": f"{filename}#{i}", "text": piece})
#
# --- Experiment B: fixed-size chunking with overlap ---
# CHUNK_SIZE = 1100
# OVERLAP = 150
# def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
#     pieces = []
#     start = 0
#     while start < len(text):
#         pieces.append(text[start:start + size])
#         start += size - overlap
#     return pieces
# for filename in os.listdir("data"):
#     if filename.endswith(".md"):
#         text = clean(open(f"data/{filename}").read())
#         for i, piece in enumerate(chunk_text(text)):
#             piece = piece.strip()
#             if piece:
#                 chunks.append({"source": f"{filename}#{i}", "text": piece})