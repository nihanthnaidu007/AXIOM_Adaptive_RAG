# The RAGAS Evaluation Framework

RAGAS (Retrieval Augmented Generation Assessment) is an evaluation framework
with reference-free metrics for RAG pipelines.

RAGAS uses embeddings internally for answer relevancy, measuring how close
generated answers sit to the question in semantic vector space, and an LLM
judge for faithfulness.

Because several RAGAS metrics are vector-based, the framework is sensitive to
the embedding model in use. RAGAS composite scores summarize faithfulness,
answer relevancy, and context groundedness in the evaluation report.
