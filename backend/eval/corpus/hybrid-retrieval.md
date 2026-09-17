# Hybrid Retrieval

Hybrid retrieval runs lexical BM25 search and dense vector search in
parallel and fuses the results, typically with reciprocal rank fusion.

The two legs are complementary: BM25 gives lexical precision on exact terms
and rare entities, while dense semantic search recalls conceptually related
passages that share no keywords with the query.

Hybrid retrieval outperforms single-strategy search because neither leg
alone covers both precision and recall; the fused precision benefit is
largest on technical corpora where exact terminology and paraphrase both
matter.
