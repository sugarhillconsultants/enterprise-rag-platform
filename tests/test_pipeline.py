"""
tests/test_pipeline.py

Formal test suite mirroring every manually-verified scenario from
development — eval metrics, BM25, vector store, hybrid fusion, chunking,
and the reranker. Every assertion here was independently confirmed
correct during development (see docs/incidents.md for the two real
findings — a chunking packing quirk and RRF's consensus-favoring
behavior — that came directly from writing these tests).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.metrics import recall_at_k, reciprocal_rank, mean_reciprocal_rank, ndcg_at_k, evaluate_retrieval
from retrieval.bm25 import BM25, tokenize
from retrieval.vector_store import NumpyVectorStore
from retrieval.hybrid import reciprocal_rank_fusion, hybrid_search
from retrieval.reranker import MockReranker
from ingestion.chunker import chunk_document, split_into_sentences
from ingestion.embeddings import TfidfEmbedder


# --- eval/metrics.py ---

def test_recall_at_k_perfect_match():
    assert recall_at_k(["a", "b", "c"], {"a"}, k=1) == 1.0

def test_recall_at_k_not_found_in_k():
    assert recall_at_k(["b", "c", "a"], {"a"}, k=2) == 0.0
    assert recall_at_k(["b", "c", "a"], {"a"}, k=3) == 1.0

def test_reciprocal_rank_and_mrr():
    all_retrieved = [["a", "b"], ["x", "a", "b"], ["x", "y"]]
    all_relevant = [{"a"}, {"a"}, {"a"}]
    assert mean_reciprocal_rank(all_retrieved, all_relevant) == (1.0 + 0.5 + 0.0) / 3

def test_ndcg_rewards_rank_position():
    assert ndcg_at_k(["a", "b", "c"], {"a"}, 3) == 1.0
    assert ndcg_at_k(["b", "c", "a"], {"a"}, 3) < 1.0

def test_evaluate_retrieval_report_shape():
    report = evaluate_retrieval([["a", "b"]], [{"a"}], k_values=[1, 2])
    assert report["n_queries"] == 1
    assert "recall@1" in report and "recall@2" in report


# --- retrieval/bm25.py ---

def test_bm25_ranks_relevant_doc_above_irrelevant():
    docs = [
        "Enable CloudTrail logging for audit compliance",
        "Quarterly sales grew in the northeast region",
    ]
    bm25 = BM25().fit(docs)
    results = bm25.search("CloudTrail logging", top_k=2)
    assert results[0][0] == 0

def test_bm25_unseen_query_terms_score_zero_not_crash():
    bm25 = BM25().fit(["some document text"])
    results = bm25.search("completely unrelated terms", top_k=1)
    assert results[0][1] == 0.0


# --- retrieval/vector_store.py ---

def test_vector_store_identical_vector_scores_near_one():
    store = NumpyVectorStore()
    store.add(np.array([[1.0, 0.0], [0.0, 1.0]]), ids=["a", "b"])
    results = store.search(np.array([1.0, 0.0]), top_k=2)
    assert results[0][0] == "a"
    assert abs(results[0][1] - 1.0) < 1e-6

def test_vector_store_mismatched_lengths_raises():
    store = NumpyVectorStore()
    try:
        store.add(np.array([[1.0, 0.0], [0.0, 1.0]]), ids=["only_one"])
        assert False, "should have raised"
    except ValueError:
        pass

def test_vector_store_empty_search_returns_empty():
    store = NumpyVectorStore()
    assert store.search(np.array([1.0, 0.0]), top_k=5) == []


# --- retrieval/hybrid.py ---

def test_rrf_agreement_wins():
    fused = reciprocal_rank_fusion([[5, 2, 1], [5, 3, 1]])
    assert fused[0][0] == 5

def test_rrf_consensus_beats_single_method_top_rank():
    """See docs/incidents.md — this documents RRF's real, non-obvious
    consensus-favoring behavior with standard k=60, found while writing
    this exact test with an initially incorrect expectation."""
    bm25_ranking = [1, 2, 3]
    vector_ranking = [7, 1, 2]
    fused = reciprocal_rank_fusion([bm25_ranking, vector_ranking])
    assert fused[0][0] == 1  # appears in both lists, beats doc 7 (top-1 in only one list)

def test_hybrid_search_top_k_truncation():
    result = hybrid_search([1, 2, 3], [3, 2, 1], top_k=2)
    assert len(result) == 2


# --- ingestion/chunker.py ---

def test_short_document_single_chunk():
    chunks = chunk_document("A short policy statement.", source_id="doc1", max_chunk_chars=500)
    assert len(chunks) == 1

def test_long_document_no_data_loss_despite_packing_quirk():
    """See docs/incidents.md — chunk sizes aren't perfectly optimal
    after the first chunk, but no sentence is ever lost."""
    text = " ".join([f"Sentence number {i} about a security control." for i in range(15)])
    chunks = chunk_document(text, source_id="doc2", max_chunk_chars=150, overlap_sentences=1)
    all_chunked_sentences = set()
    for c in chunks:
        all_chunked_sentences.update(split_into_sentences(c.text))
    assert all_chunked_sentences == set(split_into_sentences(text))

def test_empty_document_no_chunks():
    assert chunk_document("", source_id="doc3") == []


# --- ingestion/embeddings.py ---

def test_tfidf_embedder_requires_fit_before_embed():
    embedder = TfidfEmbedder()
    try:
        embedder.embed(["some text"])
        assert False, "should have raised"
    except RuntimeError:
        pass

def test_tfidf_embedder_produces_correct_shape():
    # NOTE: sklearn's TfidfVectorizer default token pattern requires
    # 2+ characters per token, so single-letter words (e.g. "a b c")
    # get filtered out entirely, producing an empty vocabulary and a
    # ValueError. Found by writing this exact test with single-letter
    # fixture data — a test-fixture bug, not a bug in TfidfEmbedder
    # itself. Using realistic multi-character words here instead.
    embedder = TfidfEmbedder(max_features=50).fit(["cat dog bird", "dog bird fish"])
    vectors = embedder.embed(["cat dog bird", "dog bird fish"])
    assert vectors.shape[0] == 2


# --- retrieval/reranker.py ---

def test_mock_reranker_reorders_by_term_overlap():
    reranker = MockReranker()
    candidates = [("low", "irrelevant text"), ("high", "unauthorized access detection")]
    results = reranker.rerank("unauthorized access detection", candidates, top_k=2)
    assert results[0].doc_id == "high"

def test_mock_reranker_top_k_truncation():
    reranker = MockReranker()
    candidates = [("a", "x"), ("b", "y"), ("c", "z")]
    results = reranker.rerank("x", candidates, top_k=1)
    assert len(results) == 1
