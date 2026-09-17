# Dense and Sparse Embeddings

Dense embeddings represent text as fixed-size vectors where every dimension
contributes, capturing semantic meaning beyond keyword overlap. Sparse
representations like BM25 vectors are mostly zeros: each dimension maps to a
vocabulary term.

Dense embeddings generalize across paraphrases — semantically similar
sentences have similar vectors even when they share no words — while sparse
representations match exact lexical items.

Modern retrieval systems combine both: sparse representations give precision
on rare terms, dense embeddings give semantic recall.
