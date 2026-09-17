# Cross-Encoder Reranking

Cross-encoder reranking is the second retrieval stage. After the initial
retrieval step — BM25, vector, or hybrid — produces a candidate list, a
cross-encoder scores each candidate jointly with the query and produces a
fine-grained relevance score.

Reranking improves precision because the cross-encoder sees query-document
interaction that the independent bi-encoder misses.

The reranked ordering determines which chunks the answer generator reads.
