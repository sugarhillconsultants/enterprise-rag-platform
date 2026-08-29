# Enterprise RAG Platform

The fifth project in this MLOps/LLM portfolio — the first to cover
retrieval-augmented generation: ingestion, chunking, hybrid retrieval
(BM25 + vector search via Reciprocal Rank Fusion), reranking, and —
the piece most tutorial-level RAG projects skip — actual retrieval
evaluation (recall@k, MRR, nDCG).

Framed around this portfolio's recurring security-analyst use case: a
knowledge base of security policy documents and incident reports,
queryable in natural language.

## Status: core logic fully verified, production embedding path unverified — stated honestly

This project was built with no network access available in
development. Every algorithmic component (BM25, vector similarity,
hybrid fusion, chunking, evaluation metrics, reranking mechanics) is
implemented, tested, and verified — **20/20 tests passing**, including
three genuine findings discovered while writing those tests (see
[`docs/incidents.md`](docs/incidents.md)). The real semantic embedding
model and cross-encoder reranker are written correctly against their
libraries' documented APIs but have **not** been executed, since that
requires network access this environment doesn't have. Full honest
breakdown: [`docs/architecture.md`](docs/architecture.md).

## What's actually in this repo

| Path | What it does | Verified? |
|---|---|---|
| `eval/metrics.py` | Recall@k, precision@k, MRR, nDCG | **Yes** — hand-computed expected values confirmed exactly |
| `retrieval/bm25.py` | BM25 keyword search, implemented from scratch | **Yes** — correctly ranks relevant docs above irrelevant ones, handles unseen terms gracefully |
| `retrieval/vector_store.py` | Cosine similarity search (numpy) + FAISS production path | Numpy backend: **yes**, exact similarity scores confirmed. FAISS backend: correct code, unverified |
| `retrieval/hybrid.py` | Reciprocal Rank Fusion combining BM25 + vector rankings | **Yes** — including a real, documented finding about RRF's consensus-favoring behavior |
| `retrieval/reranker.py` | Reranking (mock + real cross-encoder path) | Mock: **yes**, exact scores confirmed. Cross-encoder: correct code, unverified |
| `ingestion/chunker.py` | Sentence-aware document chunking with overlap | **Yes** — confirmed zero data loss despite a documented packing-efficiency quirk |
| `ingestion/embeddings.py` | TF-IDF (verified stand-in) + sentence-transformer (real, unverified) | TF-IDF: **yes**. Sentence-transformer: correct code, unverified |
| `app/main.py` | FastAPI service tying it all together | Endpoint logic verified directly (FastAPI itself not installed in dev environment) |
| `tests/test_pipeline.py` | Formal test suite — 20 tests, all passing | **Yes** — run manually (pytest unavailable in dev sandbox) |
| `docs/architecture.md` | The full honest verified/unverified breakdown | — |
| `docs/incidents.md` | 4 real findings from building and testing this project | — |

## Why hybrid retrieval, not just vector search

Vector search alone misses exact-match terms (CVE numbers, specific
policy names, error codes) that don't need semantic understanding —
just an exact match. Keyword search alone misses semantically related
content that uses different vocabulary. This project fuses both
rankings via Reciprocal Rank Fusion rather than picking one — and
`docs/incidents.md` documents a real, concrete case proving why the
semantic half of that fusion matters (a TF-IDF-only pipeline was fooled
by keyword rarity into missing the actually-relevant documents).

## Running it yourself

```bash
pip install -r app/requirements.txt
uvicorn app.main:app --reload --port 7860

# Ingest a document
curl -X POST http://localhost:7860/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_id": "cloudtrail-policy", "text": "Enable CloudTrail logging in all AWS regions to detect unauthorized API activity."}'

# Query it
curl -X POST http://localhost:7860/query \
  -H "Content-Type: application/json" \
  -d '{"query": "CloudTrail unauthorized activity", "top_k": 3}'

# Run retrieval evaluation against a known ground truth
curl -X POST http://localhost:7860/eval \
  -H "Content-Type: application/json" \
  -d '{"queries": ["CloudTrail unauthorized activity"], "relevant_chunk_ids": [["cloudtrail-policy::0"]], "k_values": [1, 3]}'
```

## What I'd add next (the real, prioritized list)

1. **Deploy somewhere with network access and swap in the real
   embedding/reranking models** — the single biggest gap, per
   `docs/architecture.md`. Run the real vs. TF-IDF evaluation
   comparison and document what it actually shows.
2. Ingest the real security-document corpus this project is designed
   around (`AYI-NEDJIMI/cloud-security-en`, plus flagged events from
   [Log Anomaly Detection Platform](https://github.com/sugarhillconsultants/log-anomaly-platform)'s
   database) instead of the small illustrative examples used for testing.
3. Persist the document index (currently in-memory) and add proper
   auth, matching this portfolio's other deployed services.
4. Deploy via the same test-gated, dual-cloud CI/CD pattern as the
   rest of this portfolio, and document whatever real issues surface —
   consistent with every other project here.
