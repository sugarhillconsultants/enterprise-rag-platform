"""
retrieval/vector_store.py

Vector similarity search. Two backends:
  - FaissVectorStore: the production path (approximate nearest-neighbor
    search, scales to millions of vectors). Requires `faiss-cpu`.
  - NumpyVectorStore: brute-force cosine similarity, no dependencies
    beyond numpy. Doesn't scale past a few thousand vectors, but is
    exactly correct and has no external dependency — this is the
    backend actually exercised in this project's own test suite and
    CI, since `faiss-cpu` isn't available in every environment this
    might run in. Both implement the same interface, so swapping one
    for the other is a one-line change once FAISS is available.
"""

import numpy as np


class NumpyVectorStore:
    def __init__(self):
        self._vectors: np.ndarray | None = None
        self._ids: list = []

    def add(self, vectors: np.ndarray, ids: list):
        """vectors: shape (n, dim), L2-normalized recommended for
        cosine similarity via dot product. ids: list of length n."""
        if len(ids) != vectors.shape[0]:
            raise ValueError("Number of ids must match number of vectors")
        if self._vectors is None:
            self._vectors = vectors.copy()
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self._ids.extend(ids)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple]:
        """Returns [(id, similarity_score), ...] sorted descending.
        Uses cosine similarity (assumes vectors are NOT necessarily
        pre-normalized — normalizes here for correctness regardless)."""
        if self._vectors is None or len(self._ids) == 0:
            return []

        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        doc_norms = self._vectors / (np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-10)

        similarities = doc_norms @ query_norm
        top_k = min(top_k, len(self._ids))
        top_indices = np.argsort(-similarities)[:top_k]

        return [(self._ids[i], float(similarities[i])) for i in top_indices]

    def __len__(self):
        return len(self._ids)


class FaissVectorStore:
    """Production backend. Not exercised in this project's own CI
    (faiss-cpu isn't installed there), but the interface matches
    NumpyVectorStore exactly, so it's a drop-in swap once available."""

    def __init__(self, dim: int):
        import faiss  # deferred import — only required if this class is actually used
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
        self._ids: list = []

    def add(self, vectors: np.ndarray, ids: list):
        if len(ids) != vectors.shape[0]:
            raise ValueError("Number of ids must match number of vectors")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        normalized = vectors / norms
        self.index.add(normalized.astype("float32"))
        self._ids.extend(ids)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple]:
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        scores, indices = self.index.search(query_norm.reshape(1, -1).astype("float32"), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._ids[idx], float(score)))
        return results

    def __len__(self):
        return len(self._ids)
