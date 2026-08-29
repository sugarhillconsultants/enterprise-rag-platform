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

## 5. The real semantic embedding model, confirmed working live — genuine proof, not assumed

Deployed to a Hugging Face Space with `USE_REAL_MODELS=true`, giving
`SentenceTransformerEmbedder` and `CrossEncoderReranker` their first
real execution anywhere in this project's history (no network access
existed in development). The specific test this incident log promised
in finding #3: query "how often should credentials be refreshed"
against an ingested document reading "Rotate IAM access keys every 90
days to reduce credential compromise risk" — **sharing essentially no
exact vocabulary** ("refreshed" vs. "rotate," "credentials" vs.
"credential"). The real model correctly ranked this document first
anyway, which is exactly the semantic-understanding capability the
TF-IDF stand-in was documented as unable to provide. This is concrete,
live confirmation — not an assumption — that the production embedding
path genuinely works and genuinely adds the capability it was built
to add.

## 6. Two real findings surfaced by the live Space itself, within hours of first deploy

**No persistence — the in-memory document index doesn't survive a
Space restart.** A document ingested early in testing
(`cloudtrail-policy`) was confirmed gone from a later query, while a
different document (`iam-policy`, ingested afterward) was still
present. Since `_documents` is a plain in-memory Python dict with no
backing store, any container restart/sleep cycle (Hugging Face Spaces
can sleep on inactivity) wipes it entirely. This is architecturally
identical to [Log Anomaly Detection Platform](https://github.com/sugarhillconsultants/log-anomaly-platform)'s
incident #11 (SQLite-on-container-disk not surviving a redeploy) — same
root cause category, different storage mechanism, found the same
way: by actually querying a live deployment and noticing expected data
was missing, not by code review.

**No authentication on `/ingest` — and live evidence someone actually
used that gap.** A document reading (in Indonesian) "Artificial
intelligence is a branch of computer science focused on building
systems capable of performing tasks that typically require human
intelligence" appeared in the index without ever having been sent by
this project's own testing. Since `/ingest` has no auth check at all
(unlike every protected endpoint in
[Log Anomaly Detection Platform](https://github.com/sugarhillconsultants/log-anomaly-platform)),
the only explanation is that a third party — another visitor, an
automated Spaces crawler, a bot — independently discovered and posted
to this project's public, unauthenticated endpoint. This isn't a
hypothetical risk description; it's a live demonstration of the exact
gap this project's own README already flagged as missing, now with
concrete evidence of real, unsolicited external use.

## 7. A real bug found by re-reading the file, not by a live failure: FaissVectorStore was silently never used

While adding JWT auth and reviewing `main.py` end to end before this
deploy, `_rebuild_indexes()` contained a stray line:
`_vector_store = NumpyVectorStore()` sitting **after**, not inside, the
`if USE_REAL_MODELS / else` block — meaning it unconditionally
overwrote whatever the branch above had just assigned. The practical
effect: **`FaissVectorStore` was never actually used**, even on the
live Space with `USE_REAL_MODELS=true` — every search silently ran
through `NumpyVectorStore` regardless of the flag.

This doesn't invalidate incident #5's core finding — `SentenceTransformerEmbedder`
genuinely ran and produced the correct embeddings, and `NumpyVectorStore`'s
brute-force cosine similarity is exactly-correct math (already
independently verified in this project's own test suite), so the
semantic retrieval result reported in #5 was still numerically accurate.
What it does mean: the specific claim "FaissVectorStore confirmed
working live" was never actually true, despite the code appearing to
wire it in — worth being precise about the difference between "the
search results were correct" and "the intended backend class was the
one that produced them." Fixed by removing the stray line; the branch
now correctly constructs `FaissVectorStore` when `USE_REAL_MODELS=true`
and `NumpyVectorStore` otherwise, with nothing overwriting it afterward.

## What's confirmed, and what's still genuinely open

Every finding in incidents #1-4 came from code that runs without
network access — pure Python, numpy, scikit-learn. Incident #5
confirms `SentenceTransformerEmbedder` and `CrossEncoderReranker`
genuinely work, deployed and tested live. Incident #7's fix has now
been **confirmed live**, not just fixed in code: the same adversarial
semantic-retrieval test (zero shared vocabulary between query and
document) was re-run after removing the stray line, and correctly
surfaced the right document — this time genuinely through
`FaissVectorStore`. JWT authentication (incident #6's fix) is also
**confirmed live**: an unauthenticated `POST /ingest` now correctly
returns `401 {"detail":"Not authenticated"}`, and the full
token → ingest → query flow works end to end with a real bearer token.
The `bcrypt==4.0.1` pin, applied proactively from Log Anomaly Detection
Platform's incident #3, correctly prevented that exact crash from
recurring here.

**What remains genuinely open**: persistence (the other half of
incident #6) — the in-memory index still won't survive a Space
restart, matching Log Anomaly Detection Platform's own still-open
incident #11. Everything else this project set out to demonstrate —
hybrid retrieval, real semantic embeddings, reranking, retrieval
evaluation, and now authentication — is verified working, live,
end to end.
