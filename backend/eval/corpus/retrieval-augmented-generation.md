# Retrieval Augmented Generation

Retrieval augmented generation (RAG) improves an LLM by retrieving relevant
documents from a knowledge base at question time and conditioning generation
on them.

If the knowledge base is missing the required information, the pipeline
should say so or fall back — a retrieval fallback, such as web search,
prevents confident answers grounded in nothing.

The purpose of AI retrieval systems is to make generation grounded and
verifiable: retrieval provides the evidence, augmented context steers the
LLM, and generation stays faithful to that evidence.
