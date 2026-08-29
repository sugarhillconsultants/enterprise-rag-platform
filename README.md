# Enterprise RAG Platform

The fifth project in this MLOps/LLM portfolio — the first to cover
retrieval-augmented generation: ingestion, chunking, hybrid retrieval
(BM25 + vector search via Reciprocal Rank Fusion), reranking, and —
the piece most tutorial-level RAG projects skip — actual retrieval
evaluation (recall@k, MRR, nDCG).

Framed around this portfolio's recurring security-analyst use case: a
knowledge base of security policy documents and incident reports,
queryable in natural language.

## Status: fully verified end to end — real models, hybrid retrieval, and auth all confirmed live

Every algorithmic component was verified locally first (20/20 tests,
including three genuine findings — see
[`docs/incidents.md`](docs/incidents.md) #1-4). Deployed to a live
Hugging Face Space, this project then confirmed, live: real semantic
embeddings correctly retrieving a document sharing zero exact
vocabulary with the query (#5); a real bug where `FaissVectorStore`
was silently never used, found and fixed, then **re-confirmed live**
after the fix (#7); and JWT authentication (reusing
[Log Anomaly Detection Platform](https://github.com/sugarhillconsultants/log-anomaly-platform)'s
exact proven pattern, including a proactive fix for that project's
known `bcrypt` incompatibility) closing a real gap that an unknown
third party had already exploited within hours of first deploy (#6).
The one remaining open item: persistence — the in-memory index still
won't survive a Space restart, the same architectural gap
[documented](https://github.com/sugarhillconsultants/log-anomaly-platform/blob/main/docs/incidents.md)
(and not yet closed) in that same companion project.

## What's actually in this repo

| Path | What it does | Verified? |
|---|---|---|
| `eval/metrics.py` | Recall@k, precision@k, MRR, nDCG | **Yes** — hand-computed expected values confirmed exactly |
| `retrieval/bm25.py` | BM25 keyword search, implemented from scratch | **Yes** — correctly ranks relevant docs above irrelevant ones, handles unseen terms gracefully |
| `retrieval/vector_store.py` | Cosine similarity search (numpy) + FAISS production path | Numpy backend: **yes**, exact similarity scores confirmed. FAISS backend: correct code, unverified |
| `retrieval/hybrid.py` | Reciprocal Rank Fusion combining BM25 + vector rankings | **Yes** — including a real, documented finding about RRF's consensus-favoring behavior |
| `retrieval/reranker.py` | Reranking (mock + real cross-encoder path) | Mock: **yes**, exact scores confirmed. Cross-encoder: **confirmed working live** on the deployed Space |
| `ingestion/chunker.py` | Sentence-aware document chunking with overlap | **Yes** — confirmed zero data loss despite a documented packing-efficiency quirk |
| `ingestion/embeddings.py` | TF-IDF (verified stand-in) + sentence-transformer (real) | Both **confirmed working** — sentence-transformer verified live on the deployed Space |
| `app/main.py` | FastAPI service tying it all together | Endpoint logic verified directly (FastAPI itself not installed in dev environment) |
| `tests/test_pipeline.py` | Formal test suite — 20 tests | **Yes, twice over**: manually verified during development (pytest unavailable in that sandbox), then confirmed passing under real `pytest` in GitHub Actions CI on first push |
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

1. **Add persistence** — the single remaining open item. The in-memory
   document index doesn't survive a Space restart (confirmed live in
   incident #6), the same architectural gap Log Anomaly Detection
   Platform still has open too. JWT auth (the other half of that
   incident) is now fixed and confirmed live.
2. Run the real-embeddings-vs-TF-IDF evaluation comparison (`POST
   /eval` against both backends on the same query set) and document
   the actual numbers — the qualitative proof already exists (incident
   #5), a quantitative comparison would be a strong addition.
3. Ingest the real security-document corpus this project is designed
   around (`AYI-NEDJIMI/cloud-security-en`, plus flagged events from
   Log Anomaly Detection Platform's database) instead of the small
   illustrative examples used for testing and this deploy.
4. Deploy via the same test-gated, dual-cloud CI/CD pattern as the
   rest of this portfolio (Azure Container Apps alongside the Hugging
   Face Space), and document whatever real issues surface — consistent
   with every other project here.
