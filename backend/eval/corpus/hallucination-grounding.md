# Hallucination Detection and Grounding

A hallucination in an LLM is generated content not supported by the
retrieved context. Faithfulness scoring detects hallucination by checking
whether claims in the answer are grounded in the retrieved chunks; low
faithfulness triggers correction in self-correcting pipelines.

Reduction techniques include grounding answers in citations, requiring the
model to answer only from the provided context, and rejecting or regenerating
low-faithfulness answers.

Grounding every claim is what makes RAG outputs trustworthy.
