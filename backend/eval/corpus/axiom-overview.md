# The AXIOM System

AXIOM is an adaptive RAG (retrieval augmented generation) pipeline. The
system classifies each query and routes it to BM25, vector, or hybrid
retrieval, reranks candidates, generates a grounded answer, and evaluates
the answer with a RAGAS critic; low-faithfulness answers trigger
self-correction.

The retrieval pipeline stores traces, chunk embeddings, and evaluation runs
in PostgreSQL, and the semantic cache is backed by Redis.

A summary of the important key components: the classifier, the retrieval
strategies, the reranker, the critic, and the observability tables. The
AXIOM approach's main advantage, in comparison with single-strategy
pipelines, is adaptive routing with self-correction. For an overview of
retrieval quality, the evaluation dashboard reports quality and evaluation
metrics for every run, and the knowledge base holds the information the
pipeline retrieves.
