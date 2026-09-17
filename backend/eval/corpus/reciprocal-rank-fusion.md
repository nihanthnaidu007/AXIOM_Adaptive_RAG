# Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) combines multiple ranked lists into a single
fused ranking. The RRF score for each document is the sum over retrieval
legs of 1 / (k + rank), where k is a constant, conventionally 60.

Because RRF uses only ranks, it fuses BM25 lexical results and dense vector
results without any score normalization. Reciprocal rank fusion is the
standard fusion step in hybrid retrieval pipelines; the k constant dampens
the influence of the very top ranks so one leg cannot dominate the fused
score.
