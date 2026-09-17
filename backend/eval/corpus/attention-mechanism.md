# The Attention Mechanism

The attention mechanism relates queries to keys and values, and this
relevance structure is the core of transformer models. In retrieval terms,
attention computes a relevance score between a query vector and each key
vector, then aggregates the corresponding values.

Self-attention lets every token attend to every other token, which is why
transformers capture long-range dependencies. The same query, key, value
pattern is why transformer representations power modern embedding models for
information retrieval.
