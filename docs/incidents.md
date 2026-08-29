# Real Findings From Building This Project

Same rationale as every other project in this portfolio: an honest
account of what was actually discovered while writing and testing this
code, not a cleaned-up version of events. All three findings below
came directly from writing tests with an initial assumption that turned
out to be wrong — in each case, the fix was to correct the assumption
(and document the real behavior), not to force the test to agree with
a mistaken expectation.

## 1. Chunking: a real (if minor) packing-efficiency quirk, no data loss

Testing `chunk_document()` on a 20-sentence document revealed that
chunk sizes shrink after the first chunk (from ~3 sentences per chunk
down to ~2), even though 3 sentences would still fit under the
character limit. Root cause: the overlap-carry step resets the buffer
to the last N sentences of the just-flushed chunk, then reprocesses
the triggering sentence against that smaller buffer — this settles
into a steady state of smaller chunks rather than repacking optimally.
Verified this causes **zero data loss** (every sentence still appears
in at least one, usually two, chunks) — it's a chunk-size efficiency
quirk, not a correctness bug. Documented in the code rather than fixed,
since correctness (no lost content, working overlap) is what actually
matters for retrieval quality; chunk-size optimality is a secondary
tuning concern worth revisiting if it becomes a real issue.

## 2. Reciprocal Rank Fusion: a genuine, non-obvious algorithmic property

Wrote a test expecting a document that ranks #1 in vector search (but
has zero keyword overlap, so it's absent from BM25 entirely) to surface
near the top of the fused hybrid ranking. It didn't — a different
document, present in *both* rankings even at moderate positions, won
instead. This is not a bug: with RRF's standard smoothing constant
(k=60), the algorithm is specifically designed to reward **consensus
across multiple retrieval methods** over a single method's top pick,
because the `+60` offset heavily dampens any one method's raw rank
contribution. This is a well-documented, intentional property of RRF
(it's what makes it more robust than naively unioning each method's
top results) — but it means a genuinely relevant document that only
one retrieval method finds may rank lower in the fused results than
intuition suggests. Fixed the test's expectation (not the code), and
documented this property prominently in `hybrid.py`'s docstring, since
it's exactly the kind of thing someone tuning `k` for a real deployment
needs to understand rather than discover by accident.

## 3. TF-IDF being fooled by term rarity — real, concrete proof of an already-documented limitation

`embeddings.py` already warned that TF-IDF vectors aren't real semantic
embeddings. Testing the full pipeline against the query "unauthorized
access detection" produced concrete proof: a document about IAM key
*rotation* ranked above two documents actually about detecting
unauthorized activity, because "access" was a rarer term in the tiny
test corpus than "unauthorized" (which both truly-relevant documents
shared), so TF-IDF over-weighted it. This is a textbook demonstration
of keyword-based retrieval being fooled by statistical rarity rather
than meaning — exactly the failure mode a real semantic embedding
model (`SentenceTransformerEmbedder`) is designed to avoid. Rather than
picking an easier test query to make this pass, this exact example is
preserved here as concrete evidence for why the production embedding
path matters, not just an abstract caveat in a docstring.

## 4. A test-fixture bug, not a code bug: single-letter tokens vanish under sklearn's default tokenizer

`TfidfEmbedder`'s own test initially used single-letter fixture words
("a b c", "b c d") and failed with `ValueError: empty vocabulary`.
Cause: `sklearn`'s `TfidfVectorizer` default token pattern requires 2+
characters per token, silently filtering out every single-letter
"word" in the fixture, leaving nothing to build a vocabulary from. The
embedder itself is correct; the test's fixture data was unrealistic.
Fixed by using real multi-character words instead.

## What's still honestly unverified

Every finding above came from code that actually runs in this
project's development environment — pure Python, numpy, and scikit-learn,
no network required. `SentenceTransformerEmbedder` and
`CrossEncoderReranker` (the real production embedding and reranking
models) have **not** been executed here at all — no network access to
download model weights. The code is written correctly against each
library's documented API, but per this portfolio's own standard,
"the code looks right" is not the same claim as "this has been run and
confirmed working." See `docs/architecture.md` for exactly what's
needed to close that gap.
