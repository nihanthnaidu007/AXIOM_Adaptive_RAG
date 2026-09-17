# Document Chunking Strategies

Chunking a document splits it into retrieval-sized pieces. Strategies
include fixed-size chunking with overlap, sentence-aware chunking, and
semantic chunking that splits on topic boundaries.

The chunking strategy matters because retrieval quality depends on chunk
granularity: chunks that are too large dilute relevance, chunks that are too
small lose surrounding context.

Overlap between consecutive chunks preserves sentences and ideas that
straddle chunk boundaries.
