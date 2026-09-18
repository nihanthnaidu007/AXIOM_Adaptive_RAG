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

    async def index_run(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Batch-write chunks from one connector run (many sources, one batch).

        The batch boundary is the RUN, not the document: one embedding call,
        one BM25 removal+rebuild per source but a single pass overall, and —
        at the caller level — ONE semantic-cache clear for the whole run.
        The per-document loop (rebuild-per-add, cache-clear-per-doc) is the
        quadratic failure shape this method exists to prevent.
        """
        if not chunks:
            return {"mode": "real", "bm25": "no_chunks", "vector": "no_chunks",
                    "chunk_count": 0, "embedding_count": 0, "sources": 0}

        texts = [c["content"] for c in chunks]
        try:
            embeddings = await embed_batch(texts)
        except Exception as exc:
            logger.error("Embedding generation failed for connector run: %s", exc)
            return {
                "mode": "real",
                "bm25": "not_attempted",
                "vector": "failed",
                "vector_error": str(exc),
                "chunk_count": len(chunks),
                "embedding_count": 0,
                "sources": len({c.get("source", "unknown") for c in chunks}),
            }

        sources = list(dict.fromkeys(c.get("source", "unknown") for c in chunks))
        by_source: Dict[str, List[Dict]] = {}
        embeddings_by_source: Dict[str, List[List[float]]] = {}
        for chunk, emb in zip(chunks, embeddings):
            source = chunk.get("source", "unknown")
            by_source.setdefault(source, []).append(chunk)
            embeddings_by_source.setdefault(source, []).append(emb)

        inserted = 0
        vector_error = None
        for source, source_chunks in by_source.items():
            await bm25_index.remove_source(source)
            await bm25_index.add_chunks(source_chunks)
            try:
                inserted += await vector_store.replace_by_source(
                    source, source_chunks, embeddings_by_source[source]
                )
            except Exception as exc:
                logger.error("pgvector replace failed for connector source %s: %s", source, exc)
                vector_error = str(exc)

        result = {
            "mode": "real",
            "bm25": "indexed",
            "vector": "failed" if vector_error else "indexed",
            "chunk_count": len(chunks),
            "embedding_count": len(embeddings),
            "sources": len(sources),
            "rows_inserted": inserted,
        }
        if vector_error:
            result["vector_error"] = vector_error
        return result


_indexer = None


def get_dual_indexer():
    global _indexer
    if _indexer is None:
        _indexer = DualIndexer()
    return _indexer
