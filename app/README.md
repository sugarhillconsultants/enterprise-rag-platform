---
title: Enterprise RAG Platform
emoji: 📚
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Enterprise RAG Platform — Security Knowledge Base

Hybrid retrieval (BM25 + real sentence-transformer embeddings via
Reciprocal Rank Fusion) with cross-encoder reranking, deployed with
`USE_REAL_MODELS=true` — the actual semantic embedding and reranking
models, not the TF-IDF/mock stand-ins used for this project's own
offline test suite. See the parent repo's
[architecture doc](https://github.com/sugarhillconsultants/enterprise-rag-platform/blob/main/docs/architecture.md)
for the full honest breakdown of what was verified where.

## Endpoints

- `GET /` — status, including which backend (real models vs. stand-in) is active
- `GET /health` — readiness probe target
- `POST /ingest` — chunk and index a document (`source_id`, `text`)
- `POST /query` — hybrid retrieval + reranking (`query`, `top_k`)
- `POST /eval` — retrieval evaluation (recall@k, MRR, nDCG) against a supplied ground truth

## Note

This is the first real, live test of `SentenceTransformerEmbedder` and
`CrossEncoderReranker` — correct code written against documented APIs,
but never executed in this project's network-free development
environment. Expect this to surface real issues on first deploy, the
same way every other project in this portfolio did.
