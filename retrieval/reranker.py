"""
retrieval/reranker.py

Reranking: takes the top-N candidates from hybrid retrieval and
re-scores them with a model that looks at the query and each document
TOGETHER (a cross-encoder), rather than comparing independently-computed
embeddings. Cross-encoders are more accurate than embedding similarity
alone but too slow to run over an entire corpus — the standard pattern
is: fast retrieval (BM25 + vector) narrows thousands of docs to ~20-50,
then a cross-encoder reranks just those before the final top-k is
chosen. This is the step most tutorial-level RAG projects skip entirely.

Two implementations, same honest split as embeddings.py:
  - CrossEncoderReranker: the real production path (needs a pretrained
    cross-encoder model and network access to download it — NOT run in
    this project's development environment).
  - MockReranker: deterministic, testable re-scoring logic used to
    verify the RERANKING MECHANICS (does it correctly reorder by score,
    does it correctly truncate to top_k) without needing a real model.
    It does NOT validate whether reranking actually improves retrieval
    quality — that requires the real model and real evaluation data.
"""

from dataclasses import dataclass


@dataclass
class RerankResult:
    doc_id: str
    text: str
    score: float


class MockReranker:
    """Deterministic reranker for testing the reordering/truncation
    mechanics. Scores documents by a simple, predictable rule (here:
    count of query terms appearing in the doc) so test expectations
    can be hand-computed exactly, unlike a real model's opaque scores."""

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int = 5) -> list[RerankResult]:
        """candidates: list of (doc_id, text) tuples from the retrieval stage."""
        query_terms = set(query.lower().split())
        scored = []
        for doc_id, text in candidates:
            doc_terms = set(text.lower().split())
            score = float(len(query_terms & doc_terms))
            scored.append(RerankResult(doc_id=doc_id, text=text, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


class CrossEncoderReranker:
    """Production reranker — a genuine cross-encoder model. Requires
    `sentence-transformers` and network access to download weights;
    this class has correct, current code but has NOT been executed in
    this project's development environment. Verify it actually works
    once deployed somewhere with network access — don't assume it
    works just because the code looks right, per this project's own
    standard for every other component."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # deferred import
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int = 5) -> list[RerankResult]:
        pairs = [(query, text) for doc_id, text in candidates]
        scores = self.model.predict(pairs)
        results = [
            RerankResult(doc_id=doc_id, text=text, score=float(score))
            for (doc_id, text), score in zip(candidates, scores)
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
