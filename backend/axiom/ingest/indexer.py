"""AXIOM Indexer - Dual-index writer (BM25 + pgvector)."""

import logging
from typing import Any, Dict, List

from axiom.retrieval.bm25_index import bm25_index
from axiom.retrieval.embeddings import embed_batch
from axiom.retrieval.vector_store import vector_store

logger = logging.getLogger(__name__)


class DualIndexer:
    """Write chunks into both BM25 and pgvector indexes.

    Indexing a source REPLACES any previously indexed chunks for that source:
    when an updated file is re-uploaded, its stale chunks must not survive
    alongside the new set.
    """

    async def index_chunks(self, chunks: List[Dict]) -> Dict[str, Any]:
        if not chunks:
            return {"mode": "real", "bm25": "no_chunks", "vector": "no_chunks", "chunk_count": 0, "embedding_count": 0}

        source = chunks[0].get("source", "unknown")
        texts = [c["content"] for c in chunks]
        try:
            embeddings = await embed_batch(texts)
        except Exception as exc:
            logger.error("Embedding generation failed: %s", exc)
            return {
                "mode": "real",
                "bm25": "not_attempted",
                "vector": "failed",
                "vector_error": str(exc),
                "chunk_count": len(chunks),
                "embedding_count": 0,
            }

        # Embeddings succeeded — now mutate both indexes. BM25 first, then
        # pgvector, mirroring the historical ordering.
        await bm25_index.remove_source(source)
        await bm25_index.add_chunks(chunks)

        try:
            inserted = await vector_store.replace_by_source(source, chunks, embeddings)
        except Exception as exc:
            logger.error("pgvector replace failed: %s", exc)
            return {
                "mode": "real",
                "bm25": "indexed",
                "vector": "failed",
                "vector_error": str(exc),
                "chunk_count": len(chunks),
                "embedding_count": len(embeddings),
            }

        return {
            "mode": "real",
            "bm25": "indexed",
            "vector": "indexed",
            "chunk_count": len(chunks),
            "embedding_count": len(embeddings),
            "rows_inserted": inserted,
        }


_indexer = None


def get_dual_indexer():
    global _indexer
    if _indexer is None:
        _indexer = DualIndexer()
    return _indexer
