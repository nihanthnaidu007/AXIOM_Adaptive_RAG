# Comparing Retrieval Strategies

Comparing retrieval strategies: BM25 gives lexical precision and needs no
model; vector search gives semantic recall through embeddings; hybrid
retrieval fuses both with reciprocal rank fusion.

Cross-encoder reranking then reorders the fused candidates with a relevance
model, and a semantic cache skips the pipeline entirely for repeated
queries.

Each approach's advantage depends on the corpus: exact terms favor BM25,
paraphrase-heavy domains favor vector search, and the hybrid approach wins
on mixed workloads in this comparison.
