# Embedding Model Performance

Embedding model performance is tracked with retrieval benchmarks such as
MTEB. Improvements in embedding models come from better training objectives,
instruction-tuned embeddings, and larger context windows.

When benchmarking an embedding model, measure recall@k on your own corpus: a
model that tops public benchmarks may underperform on domain-specific text.

Model choice interacts with dimensionality — larger embedding models give
better semantic performance at higher storage cost.
