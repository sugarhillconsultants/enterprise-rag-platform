# Architecture: What's Verified, What Needs Network Access to Confirm

## The honest split, stated plainly up front

This project was built with **no network access** in its development
environment — no way to install `faiss-cpu`, `sentence-transformers`,
or `rank_bm25`, and no way to download a real embedding or
cross-encoder model. Rather than write code against those libraries
and simply assume it works, every component was built with two paths:

| Component | Verified locally (this repo's own tests/CI) | Production path (needs network, unverified here) |
|---|---|---|
| Keyword retrieval | `BM25` — implemented from scratch against the standard formula, since `rank_bm25` wasn't available | Same code either way — BM25 doesn't need a pretrained model |
| Vector similarity | `NumpyVectorStore` — brute-force cosine similarity, exact and dependency-free | `FaissVectorStore` — approximate nearest-neighbor, scales to millions of vectors |
| Embeddings | `TfidfEmbedder` — real vectors, but keyword-overlap-based, NOT semantic | `SentenceTransformerEmbedder` — genuine semantic embeddings, needs model download |
| Reranking | `MockReranker` — deterministic term-overlap scoring, verifies reordering/truncation mechanics only | `CrossEncoderReranker` — genuine cross-encoder relevance scoring, needs model download |
| Retrieval evaluation | `eval/metrics.py` — pure math (recall@k, MRR, nDCG), fully verified regardless of which embedder/reranker is used | Same code either way |
| Hybrid fusion | `reciprocal_rank_fusion()` — pure logic, fully verified | Same code either way |

**The evaluation engine and hybrid fusion logic are backend-agnostic** —
they work identically whether the underlying vectors come from TF-IDF
or a real sentence-transformer. This matters: once this runs somewhere
with network access, swapping `TfidfEmbedder` for
`SentenceTransformerEmbedder` and `MockReranker` for
`CrossEncoderReranker` is a small, contained change (a few lines in
`app/main.py`, gated by the existing `USE_REAL_MODELS` environment
variable) — the retrieval pipeline, fusion logic, and evaluation
metrics don't need to change at all.

## What TF-IDF actually validates, and what it doesn't

Using `TfidfEmbedder` to test this pipeline is legitimate for
validating **mechanics**: does ingestion correctly chunk and index
documents, does the vector store correctly rank by similarity, does
hybrid fusion correctly combine two rankings, does the evaluation
engine correctly compute recall/MRR/nDCG. All of that is genuinely
proven by this project's test suite.

What it does **not** validate is retrieval **quality** in the way a
real semantic embedding model would deliver it. `docs/incidents.md` #3
documents a concrete, reproducible case where TF-IDF was fooled by
keyword rarity into ranking an irrelevant document above two genuinely
relevant ones — exactly the failure mode real embeddings are designed
to avoid. Anyone evaluating this project's retrieval *quality* should
do so against the real `SentenceTransformerEmbedder` path, not the
TF-IDF stand-in.

## What it would take to close this gap for real

1. Deploy this service somewhere with network access (a Hugging Face
   Space, an Azure Container App — same pattern as this portfolio's
   other three deployed projects).
2. Install `sentence-transformers` and `faiss-cpu` from the commented-out
   lines in `app/requirements.txt`.
3. Set `USE_REAL_MODELS=true` and swap the embedder/reranker
   instantiation in `app/main.py` to the production classes.
4. **Actually run the retrieval evaluation** (`POST /eval` with a real
   query set and ground-truth relevant chunk IDs) against both the
   TF-IDF and real-embedding backends, and compare the reported
   recall@k/MRR/nDCG numbers directly — that comparison is the concrete
   proof (or disproof) of how much retrieval quality the real embedding
   model actually adds over the free, dependency-light stand-in.
5. Document whatever that comparison actually shows, the same way this
   portfolio's other projects document real experimental results
   rather than assumed ones (see Project 2's dataset-impact experiment
   for the pattern this would follow).
