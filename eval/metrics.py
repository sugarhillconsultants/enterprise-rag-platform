"""
eval/metrics.py

Retrieval evaluation — the piece most tutorial-level RAG projects skip
entirely. A RAG system that has never measured whether its retrieval
step actually finds the right documents is not meaningfully different
from a system that returns whatever it feels like; these metrics are
what make "the retriever works" a checkable claim rather than a vibe.

Standard information-retrieval metrics, computed against a set of
queries each with a known set of relevant document IDs (ground truth).
"""

import numpy as np


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of all relevant documents that appear in the top k
    retrieved results. 1.0 means every relevant doc was found."""
    if not relevant_ids:
        raise ValueError("relevant_ids must be non-empty")
    top_k = set(retrieved_ids[:k])
    found = len(top_k & relevant_ids)
    return found / len(relevant_ids)


def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of the top k retrieved results that are actually relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    found = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return found / k


def reciprocal_rank(retrieved_ids: list, relevant_ids: set) -> float:
    """1 / (rank of the first relevant result), or 0 if none found.
    The building block of Mean Reciprocal Rank."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(all_retrieved: list, all_relevant: list) -> float:
    """MRR across a set of queries. Each element of all_retrieved is a
    ranked list of doc IDs for one query; all_relevant is the matching
    set of ground-truth relevant IDs for that query."""
    if len(all_retrieved) != len(all_relevant):
        raise ValueError("all_retrieved and all_relevant must be the same length")
    scores = [reciprocal_rank(r, rel) for r, rel in zip(all_retrieved, all_relevant)]
    return float(np.mean(scores))


def dcg_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Discounted Cumulative Gain — rewards relevant results appearing
    higher in the ranking, not just present anywhere in the top k."""
    top_k = retrieved_ids[:k]
    gains = [1.0 if doc_id in relevant_ids else 0.0 for doc_id in top_k]
    discounts = [1.0 / np.log2(i + 2) for i in range(len(gains))]  # rank 0 -> log2(2)
    return float(np.sum(np.array(gains) * np.array(discounts)))


def ndcg_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Normalized DCG — DCG divided by the best-possible DCG (all
    relevant docs ranked first), giving a 0-1 score comparable across
    queries with different numbers of relevant documents."""
    actual_dcg = dcg_at_k(retrieved_ids, relevant_ids, k)
    ideal_ranking = list(relevant_ids)[:k] + ["_irrelevant_"] * max(0, k - len(relevant_ids))
    ideal_dcg = dcg_at_k(ideal_ranking, relevant_ids, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def evaluate_retrieval(all_retrieved: list, all_relevant: list, k_values: list = [1, 3, 5, 10]) -> dict:
    """Full evaluation report across a query set, at multiple k values.
    This is the function a CI job or dashboard would actually call."""
    if len(all_retrieved) != len(all_relevant):
        raise ValueError("all_retrieved and all_relevant must be the same length")

    report = {"n_queries": len(all_retrieved), "mrr": mean_reciprocal_rank(all_retrieved, all_relevant)}

    for k in k_values:
        recalls = [recall_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)]
        precisions = [precision_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)]
        ndcgs = [ndcg_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)]
        report[f"recall@{k}"] = round(float(np.mean(recalls)), 4)
        report[f"precision@{k}"] = round(float(np.mean(precisions)), 4)
        report[f"ndcg@{k}"] = round(float(np.mean(ndcgs)), 4)

    return report
