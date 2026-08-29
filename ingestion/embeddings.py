"""
ingestion/embeddings.py

Turns text into vectors for semantic search. Two implementations:

  - SentenceTransformerEmbedder: the real production path, using a
    genuine pretrained embedding model (e.g. all-MiniLM-L6-v2) that
    captures actual semantic meaning. Requires `sentence-transformers`
    and network access to download the model weights — NOT available
    in this project's development/test environment, so this class is
    correct, current code that has NOT been executed end-to-end here.

  - TfidfEmbedder: a TF-IDF-based stand-in used for local development
    and this project's own test suite. TF-IDF vectors are REAL,
    legitimate numeric vectors — they are NOT semantic embeddings.
    They capture keyword overlap, not meaning, so a query and a
    document using different words for the same concept will NOT
    match well under this embedder, the way a real sentence-transformer
    embedding would. Using this to validate the retrieval PIPELINE
    (vector store indexing, hybrid fusion, evaluation) is legitimate;
    treating its retrieval quality as representative of the production
    system's quality would not be.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    """Stand-in embedder for local testing without network access.
    See module docstring for the important caveat about what this
    does and does not validate."""

    def __init__(self, max_features: int = 2000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

    def fit(self, documents: list[str]):
        self.vectorizer.fit(documents)
        self._fitted = True
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() with the document corpus before embed()")
        vectors = self.vectorizer.transform(texts)
        return vectors.toarray()

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class SentenceTransformerEmbedder:
    """Production embedder — genuine semantic embeddings. Requires
    network access to download model weights on first use; this class
    has correct, current code but has NOT been run in this project's
    development environment (no network there). Verify this actually
    works once deployed somewhere with network access, per this
    project's own honesty standard — don't assume it works just
    because the code looks right."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # deferred import
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
