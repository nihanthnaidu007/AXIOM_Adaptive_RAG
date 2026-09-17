# RAG Evaluation Best Practices

Evaluating a RAG (retrieval augmented generation) pipeline requires metrics
beyond text overlap. RAGAS is the standard evaluation framework: faithfulness
checks that every claim in the answer is entailed by the retrieved context,
answer relevancy checks that the answer addresses the question, and context
groundedness checks that the context supports the answer.

Best practices are to build a golden benchmark dataset, track the metrics
per retrieval strategy, and set thresholds that fail the pipeline when
quality regresses.

Retrieval quality and evaluation metrics should be monitored continuously,
not only at release time.
