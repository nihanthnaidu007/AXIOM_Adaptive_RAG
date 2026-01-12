"""AXIOM Indexer - Dual-index writer (BM25 + pgvector)."""

import logging
from typing import List, Dict, Any

from axiom.retrieval.bm25_index import bm25_index
from axiom.retrieval.vector_store import vector_store
from axiom.retrieval.embeddings import embed_batch

logger = logging.getLogger(__name__)


class DualIndexer:
    """Write chunks into both BM25 and pgvector indexes."""

    async def index_chunks(self, chunks: List[Dict]) -> Dict[str, Any]:
        if not chunks:
            return {"mode": "real", "bm25": "no_chunks", "vector": "no_chunks", "chunk_count": 0, "embedding_count": 0}

        bm25_index.add_chunks(chunks)

        texts = [c["content"] for c in chunks]
        try:
            embeddings = await embed_batch(texts)
        except Exception as exc:
            logger.error("Embedding generation failed: %s", exc)
            return {
                "mode": "real",
                "bm25": "indexed",
                "vector": "failed",
                "vector_error": str(exc),
                "chunk_count": len(chunks),
                "embedding_count": 0,
            }

        try:
            inserted = await vector_store.insert_chunks(chunks, embeddings)
        except Exception as exc:
            logger.error("pgvector insert failed: %s", exc)
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
