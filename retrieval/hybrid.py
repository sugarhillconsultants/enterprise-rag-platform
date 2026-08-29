"""
retrieval/hybrid.py

Hybrid retrieval: combines BM25 (keyword) and vector (semantic)
rankings via Reciprocal Rank Fusion (RRF). This is what distinguishes
a serious RAG implementation from "just embeddings" — vector search
alone tends to miss exact-match terms (specific error codes, CVE
numbers, exact policy names) that keyword search reliably catches,
while keyword search alone misses semantically-related content that
doesn't share exact vocabulary. RRF combines both rankings without
needing the two systems' raw scores to be on comparable scales (BM25
scores and cosine similarities aren't directly comparable numbers,
but RANKS from each system are).

Reference: Cormack, Clarke, Buettcher, "Reciprocal Rank Fusion
Outperforms Condorcet and Individual Rank Learning Methods" (2009).
"""


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """
    rankings: a list of ranked-document-index lists, one per retrieval
              method (e.g. [bm25_ranking, vector_ranking]).
    k: RRF's smoothing constant (60 is the standard default from the
       original paper — dampens the influence of very high individual
       ranks so one method doesn't completely dominate).

    Returns a fused ranking: [(doc_index, fused_score), ...] sorted
    descending by fused score.

    IMPORTANT, non-obvious property (found while writing this module's
    tests, worth knowing before tuning k): with the standard k=60, RRF
    favors documents that appear in MULTIPLE ranking lists (consensus
    across retrieval methods) over a document that's uniquely ranked
    #1 in only one list. Concretely: a document at rank 0 in both BM25
    and vector search will usually outscore a document at rank 0 in
    vector search alone but absent from BM25 entirely — even though
    the second document "won" one full ranking outright. This is
    intentional (it's what makes RRF more robust than naively unioning
    each method's top picks), but it means a document only vector
    search finds (no keyword overlap at all) may rank lower in the
    fused results than intuition suggests. If recall of such
    single-method-only matches matters more than consensus robustness
    for your use case, a smaller k weights individual top ranks more
    heavily at the cost of RRF's usual robustness.
    """
    fused_scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, doc_index in enumerate(ranking):
            fused_scores[doc_index] = fused_scores.get(doc_index, 0.0) + 1.0 / (k + rank + 1)

    fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def hybrid_search(bm25_ranking: list[int], vector_ranking: list[int], top_k: int = 10, rrf_k: int = 60) -> list[int]:
    """Convenience wrapper: takes two rankings (doc indices, best
    first), fuses them, and returns just the top_k doc indices."""
    fused = reciprocal_rank_fusion([bm25_ranking, vector_ranking], k=rrf_k)
    return [doc_index for doc_index, _ in fused[:top_k]]
