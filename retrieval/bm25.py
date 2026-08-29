"""
retrieval/bm25.py

BM25 (Okapi) keyword retrieval, implemented from the standard formula
rather than depending on the `rank_bm25` package (unavailable in this
development environment, with no network to install it). This is the
keyword-search half of hybrid retrieval — vector search alone tends to
miss exact-match terms (error codes, CVE numbers, specific policy
names) that keyword search catches reliably.

Reference: Robertson & Zaragoza, "The Probabilistic Relevance
Framework: BM25 and Beyond" (2009).
"""

import re
import numpy as np
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer. A production system might use a
    proper tokenizer (spaCy, a subword tokenizer) for better handling
    of hyphenated terms, acronyms, etc. — this is intentionally simple
    and sufficient for the security-document domain this project targets."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        self.idf: dict[str, float] = {}
        self.corpus_size: int = 0

    def fit(self, documents: list[str]):
        """documents: list of raw text strings, one per document."""
        self.corpus_size = len(documents)
        self.doc_freqs = []
        self.doc_lengths = []

        doc_containing_term: Counter = Counter()

        for doc in documents:
            tokens = tokenize(doc)
            self.doc_lengths.append(len(tokens))
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            for term in freqs:
                doc_containing_term[term] += 1

        self.avg_doc_length = float(np.mean(self.doc_lengths)) if self.doc_lengths else 0.0

        # Standard BM25 IDF, with the +1 smoothing that keeps IDF
        # non-negative even for terms appearing in every document.
        self.idf = {}
        for term, n_containing in doc_containing_term.items():
            self.idf[term] = np.log(
                (self.corpus_size - n_containing + 0.5) / (n_containing + 0.5) + 1
            )
        return self

    def score(self, query: str, doc_index: int) -> float:
        query_terms = tokenize(query)
        freqs = self.doc_freqs[doc_index]
        doc_length = self.doc_lengths[doc_index]

        score = 0.0
        for term in query_terms:
            if term not in freqs:
                continue
            idf = self.idf.get(term, 0.0)
            term_freq = freqs[term]
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * doc_length / self.avg_doc_length
            )
            score += idf * numerator / denominator
        return float(score)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Returns [(doc_index, score), ...] sorted by score descending."""
        scores = [(i, self.score(query, i)) for i in range(self.corpus_size)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
