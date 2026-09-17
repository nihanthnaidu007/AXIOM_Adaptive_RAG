# BM25 Scoring

BM25 (Best Matching 25) is the Okapi ranking function used for sparse lexical
retrieval. The BM25 formula combines term frequency with inverse document
frequency (IDF) to score documents against a query.

Term frequency saturation means a term appearing many times in one document
gives diminishing returns: the saturation component prevents very long
documents from dominating the ranking. The k1 parameter controls term
frequency scaling, while the b parameter controls document length
normalization.

BM25 remains a strong baseline for keyword search in retrieval augmented
generation systems because it needs no model, only an inverted index.
