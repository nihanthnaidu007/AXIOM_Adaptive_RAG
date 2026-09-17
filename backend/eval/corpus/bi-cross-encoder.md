# Bi-Encoders and Cross-Encoders

A bi-encoder encodes a query and a document independently and compares the
embeddings with cosine similarity, which is fast enough for first-stage
retrieval over millions of vectors.

A cross-encoder concatenates the query and document and scores them jointly
through the transformer, capturing fine-grained interaction but requiring
one forward pass per pair.

That is why cross-encoders are used for reranking a shortlist produced by a
bi-encoder or BM25, and never for the initial retrieval step.
