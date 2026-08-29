"""
app/main.py

FastAPI service exposing the RAG pipeline: ingest documents, query with
hybrid retrieval + reranking, and an eval endpoint reporting retrieval
metrics against a stored ground-truth set. Set USE_REAL_MODELS=true to
use genuine semantic embeddings (sentence-transformers) and a real
cross-encoder reranker; defaults to the verified TF-IDF/mock stand-ins
otherwise. See docs/architecture.md for the full honest breakdown of
what's verified locally versus what needs network access to confirm.
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ingestion.chunker import chunk_document
from ingestion.embeddings import TfidfEmbedder, SentenceTransformerEmbedder
from retrieval.bm25 import BM25
from retrieval.vector_store import NumpyVectorStore, FaissVectorStore
from retrieval.hybrid import hybrid_search
from retrieval.reranker import MockReranker, CrossEncoderReranker
from eval.metrics import evaluate_retrieval

USE_REAL_MODELS = os.environ.get("USE_REAL_MODELS", "false").lower() == "true"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2's output dimension, only used when USE_REAL_MODELS=true

# In-memory index — a real deployment would persist this; kept simple
# and explicit here since the point of this service is demonstrating
# the retrieval pipeline, not building a full document-management system.
_documents: dict[str, str] = {}   # chunk_id -> text
_embedder = None
_vector_store = None
_bm25: BM25 | None = None

# Reranker is stateless (no corpus-specific fitting needed), so it can
# be constructed once at import time rather than rebuilt per-ingest.
_reranker = CrossEncoderReranker() if USE_REAL_MODELS else MockReranker()


def _rebuild_indexes():
    global _embedder, _vector_store, _bm25
    if not _documents:
        _embedder, _vector_store, _bm25 = None, None, None
        return

    chunk_ids = list(_documents.keys())
    texts = list(_documents.values())

    if USE_REAL_MODELS:
        _embedder = SentenceTransformerEmbedder()
        vectors = _embedder.embed(texts)
        _vector_store = FaissVectorStore(dim=EMBEDDING_DIM)
    else:
        _embedder = TfidfEmbedder().fit(texts)
        vectors = _embedder.embed(texts)
        _vector_store = NumpyVectorStore()
    _vector_store = NumpyVectorStore()
    _vector_store.add(vectors, ids=chunk_ids)
    _bm25 = BM25().fit(texts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Enterprise RAG Platform — Security Knowledge Base", lifespan=lifespan)


class IngestRequest(BaseModel):
    source_id: str
    text: str


class IngestResponse(BaseModel):
    source_id: str
    chunks_created: int


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    candidates_before_rerank: int = 20


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]
    backend: str


@app.get("/")
def read_root():
    return {
        "message": "Enterprise RAG Platform is running",
        "backend": "real (sentence-transformers + cross-encoder)" if USE_REAL_MODELS else "verified local stand-in (TF-IDF + mock reranker)",
        "documents_indexed": len(_documents),
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ingest", response_model=IngestResponse, status_code=201)
def ingest(payload: IngestRequest):
    chunks = chunk_document(payload.text, source_id=payload.source_id)
    for chunk in chunks:
        chunk_id = f"{chunk.source_id}::{chunk.chunk_index}"
        _documents[chunk_id] = chunk.text

    _rebuild_indexes()
    return IngestResponse(source_id=payload.source_id, chunks_created=len(chunks))


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    if not _documents or _vector_store is None or _bm25 is None:
        raise HTTPException(status_code=400, detail="No documents ingested yet — call /ingest first")

    chunk_ids = list(_documents.keys())

    query_vec = _embedder.embed_one(payload.query)
    vector_results = _vector_store.search(query_vec, top_k=payload.candidates_before_rerank)
    vector_ranking = [chunk_ids.index(cid) for cid, score in vector_results]

    bm25_results = _bm25.search(payload.query, top_k=payload.candidates_before_rerank)
    bm25_ranking = [idx for idx, score in bm25_results]

    fused_indices = hybrid_search(bm25_ranking, vector_ranking, top_k=payload.candidates_before_rerank)
    candidates = [(chunk_ids[i], _documents[chunk_ids[i]]) for i in fused_indices]

    reranked = _reranker.rerank(payload.query, candidates, top_k=payload.top_k)

    return QueryResponse(
        query=payload.query,
        results=[RetrievedChunk(chunk_id=r.doc_id, text=r.text, score=r.score) for r in reranked],
        backend="tfidf+mock" if not USE_REAL_MODELS else "sentence-transformers+cross-encoder",
    )


class EvalRequest(BaseModel):
    queries: list[str]
    relevant_chunk_ids: list[list[str]]  # ground truth per query
    k_values: list[int] = [1, 3, 5]


@app.post("/eval")
def run_eval(payload: EvalRequest):
    if len(payload.queries) != len(payload.relevant_chunk_ids):
        raise HTTPException(status_code=400, detail="queries and relevant_chunk_ids must be the same length")

    all_retrieved = []
    for q in payload.queries:
        result = query(QueryRequest(query=q, top_k=max(payload.k_values)))
        all_retrieved.append([r.chunk_id for r in result.results])

    all_relevant = [set(ids) for ids in payload.relevant_chunk_ids]
    report = evaluate_retrieval(all_retrieved, all_relevant, k_values=payload.k_values)
    return report
