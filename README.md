# Retrieval Benchmark: Is the Bottleneck the Model or the Retrieval?

A hands-on benchmark testing a claim I kept hearing about RAG systems: that when
AI agents give bad answers, the problem is usually **retrieval**, not the language
model. "Garbage in, garbage out." I wanted to test that instead of taking it on
faith, so I built an evaluation harness and measured several retrieval approaches
on the same set of questions over real documentation.

The corpus is technical support documentation for a retail POS system I worked on
professionally, which made it possible to write realistic support questions and
judge relevance by hand.

> **Note on the corpus:** the source documents are third-party proprietary
> documentation and are **not** included in this repository (see
> [Corpus & Licensing](#corpus--licensing)). This repo contains the harness,
> the question set, and the results — not the source content.

---

## Results

Metric is **Mean Reciprocal Rank (MRR)** over a 19-question golden set. Higher is
better; 1.0 means the correct document was ranked first every time.

| Retrieval approach | MRR |
|---|---|
| Dense, small embedding model (`all-MiniLM-L6-v2`) | 0.831 |
| Dense, larger embedding model (`bge-base-en-v1.5`) | 0.860 |
| Hand-built hybrid (dense + BM25, naive fusion) | 0.737 |
| Managed retrieval engine, semantic only | **0.896** |

Headline findings:

- **The embedding model was the lever, not chunking or keyword search.** Upgrading
  the embedding model resolved the hardest questions; a naive hybrid approach made
  things *worse*.
- **A purpose-built retrieval engine beat my best hand-tuned pipeline** (0.896 vs
  0.860) on the same questions — but the two had *different* blind spots, not
  strictly better ones.
- **The hard cases were near-duplicate documents**, not obscure questions.
  Documents that share most of their vocabulary are where retrieval quality is
  actually decided.

---

## Method

### Corpus
Six support documents covering distinct POS workflows (customer records, user
accounts, suppliers, purchase orders, inventory transfers, item setup). Several of
these overlap heavily in vocabulary — purchase orders and inventory transfers, for
example, share roughly 70% of their text — which turned out to be the whole story
(see findings).

### Golden set
19 questions, each paired with a **fingerprint**: a short string that appears in
exactly one document and nowhere else, verified programmatically against the whole
corpus. The fingerprint is what marks a retrieval "correct" — it identifies the
one document that actually answers the question.

Questions were deliberately split into two kinds:
- **Close-phrased** — worded similarly to the source text (the easy case).
- **Far-phrased** — worded the way a real user would ask, using none of the
  document's own vocabulary (the case that separates good semantic retrieval from
  keyword matching).

The set is weighted toward the near-duplicate document cluster, since that is where
retrieval approaches diverge.

### Metric
**MRR** was chosen deliberately. For a support agent that answers from the single
top result, *rank-1 precision* is what matters — a correct document ranked 2nd is
nearly as bad as not finding it. MRR rewards getting the right document to the top.
(For a research/synthesis agent that reads many results, a recall-oriented metric
would be the better choice — the metric should follow how the results get used.)

---

## Experiments & Findings

### 1. Chunking: for this corpus, don't
Tested one-chunk-per-file, splitting on document structure, and fixed-size chunking
with overlap (sweeping the size). **One-chunk-per-file won (0.831); every split
scored lower.** Splitting on document structure produced wildly uneven chunk counts
(one document shattered into far more pieces than the others and flooded the
rankings). The lesson: these documents are short and single-topic — already
"atomic" — so splitting them fragments a coherent answer instead of separating
distinct ideas. Chunking helps for long, multi-topic documents; it hurt here.

### 2. Embedding model: the actual bottleneck
Swapping `all-MiniLM-L6-v2` (384-dim) for `bge-base-en-v1.5` moved MRR from 0.831 to
0.860. More importantly, it *flipped the two hardest questions from wrong to
perfect* — far-phrased inventory-transfer questions that the smaller model had
confused with an adjacent document. This isolated the real bottleneck: **embedding
resolution on near-duplicate documents**, not chunking or keyword matching. The
smaller model literally could not tell near-twin documents apart; the larger one
mostly could.

### 3. Hybrid search: a negative result worth keeping
Added BM25 keyword search alongside dense retrieval, fused with Reciprocal Rank
Fusion. It **regressed to 0.737** — worse than dense alone. Two causes, both
diagnosed from the data:
- **BM25 length bias.** One document was far larger than the rest and contained so
  many terms that it matched almost every query, polluting the rankings.
- **Equal-weight fusion.** Giving a weaker signal an equal vote dragged down strong
  dense rankings.

A fusion-weight sweep confirmed it: MRR declined *monotonically* as BM25's weight
increased. On this corpus, at every weight, keyword search could not help. The
takeaway isn't "hybrid is bad" — it's that naive hybrid without length
normalization and tuned fusion is actively harmful, which is exactly what a managed
system is supposed to handle.

### 4. Managed retrieval engine: the head-to-head
Stood up a purpose-built retrieval engine locally (its own chunking, its own
embedding model, hybrid + reranking available) and ran the identical 19 questions
through it. **Semantic-only scored 0.896** — ahead of my best hand-tuned pipeline
(0.860). It resolved several near-duplicate cases mine missed. Notably, it also
missed a couple that mine got right: the systems have *different* residual blind
spots, not a strict ordering. This is the most honest version of the result —
better on balance, not universally.

---

## Limitations

Stated plainly, because they bound what the numbers mean:

- **Small corpus (6 documents) and a single 19-question golden set.** Enough to
  surface real patterns and establish a stable MRR, but a single question flip
  moves the score meaningfully. A production evaluation would want 50+ questions
  across more documents.
- **The golden set is hand-built and weighted toward the hard cases.** That is
  intentional (to stress fine-grained retrieval), but it means the absolute numbers
  are specific to this corpus, not a universal ranking of the tools.
- **Reranking is incomplete.** The managed engine's reranker timed out when scoring
  full document bodies — the largest document made a full-body cross-encoder pass
  exceed the query timeout. Reranking on document *titles* alone (a much cheaper but
  meaningless input) degraded results, as expected. A proper test requires pruning
  candidates before reranking or capping the reranked field length. This is the main
  open thread (see next steps).
- **Fingerprint matching is exact-substring.** Robust here because each fingerprint
  was verified unique, but it does not handle paraphrase — it checks that the right
  *document* was retrieved, not that an answer was correctly generated.

---

## What I'd Do Next

- **Finish the reranker experiment** by pruning to the top few candidates before the
  cross-encoder pass, so full-document reranking stays under the query timeout. This
  tests whether a tuned reranker resolves the near-duplicate ties that pure semantic
  retrieval leaves close.
- **Expand the golden set to 50+ questions** across more documents for a more stable,
  less corpus-specific result.
- **Run the same harness against a standardized retrieval dataset** (e.g. a BEIR
  task) to cross-check the methodology against an externally credible benchmark.
- **Add pgvector** as an explicit baseline. (Reasoning suggests it reproduces the
  dense numbers, since it is the same embeddings and cosine similarity in a
  database — but running it makes the comparison concrete rather than assumed.)

---

## Repository

```
.
├── data/                 # RMH source documents — GITIGNORED, not distributed
├── make_jsonl.py         # cleans + converts source docs to JSONL for ingestion
├── RMH_agent.py          # dense/hybrid harness: embed, retrieve, score (MRR)
├── antfly_bench.py       # runs the golden set against the managed engine, scores MRR
└── rmh-docs.jsonl        # generated corpus — GITIGNORED
```

The harness is deliberately transparent: retrieval, a confidence/scoring step, and a
manager loop that runs the golden set and reports MRR. Swapping the retrieval backend
(numpy cosine, a different embedding model, the managed engine) leaves the golden set
and scorer untouched, the evaluation is held constant while the retriever varies.

## Corpus & Licensing

The source documents are proprietary third-party POS documentation. They are used
here **only** as a private, local test corpus for a personal benchmark — not
reproduced, redistributed, or published. The `data/` directory and generated corpus
files are gitignored. This repository contains only the evaluation harness, the
question set, and the measured results. Any use of this approach against that
documentation for anything beyond private evaluation would require checking the
documentation's license terms.