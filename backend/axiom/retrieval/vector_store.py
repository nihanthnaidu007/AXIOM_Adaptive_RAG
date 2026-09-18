"""AXIOM Vector Store - Real pgvector semantic search."""

import logging
from typing import Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from axiom.config import get_config

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._connected = False

    def _build_dsn(self) -> str:
        cfg = get_config()
        return (
            f"postgresql+asyncpg://{cfg.postgres_user}:{cfg.postgres_password}"
            f"@{cfg.postgres_host}:{cfg.postgres_port}/{cfg.postgres_db}"
        )

    async def connect(self) -> bool:
        try:
            cfg = get_config()
            expected_dims = cfg.effective_embedding_dimensions
            self._engine = create_async_engine(self._build_dsn(), pool_size=5, max_overflow=10)
            async with self._engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.execute(
                    text(f"""
                    CREATE TABLE IF NOT EXISTS chunk_embeddings (
                        id SERIAL PRIMARY KEY,
                        chunk_id TEXT NOT NULL UNIQUE,
                        source TEXT NOT NULL,
                        content TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        embedding vector({expected_dims}) NOT NULL,
                        token_count INTEGER,
                        bm25_score FLOAT,
                        page_start INTEGER,
                        page_end INTEGER,
                        origin_type TEXT,
                        ingested_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                )
                await conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS chunk_embeddings_vec_idx "
                        "ON chunk_embeddings "
                        "USING ivfflat (embedding vector_cosine_ops) "
                        "WITH (lists = 100)"
                    )
                )
                # Dimension-honest startup check (W3): an existing table built
                # under a different width would silently reject every insert
                # with a cryptic pgvector error. Fail visibly instead, naming
                # the exact remediation — never truncate, never coerce.
                row = await conn.execute(
                    text(
                        "SELECT atttypmod FROM pg_attribute "
                        "WHERE attrelid = 'chunk_embeddings'::regclass "
                        "AND attname = 'embedding' AND attisdropped = false"
                    )
                )
                typmod = (row.fetchone() or [None])[0]
                actual_dims = int(typmod) if typmod is not None and typmod >= 0 else None
                if actual_dims is not None and actual_dims != expected_dims:
                    logger.error(
                        "chunk_embeddings.embedding is vector(%d) but the configured "
                        "embedding width is %d (%s). Refusing to start indexing "
                        "against mismatched vectors. Fix: run `python -m axiom.reembed "
                        "--yes` (truncates + rebuilds the column), re-upload the "
                        "listed sources, then `alembic upgrade head`.",
                        actual_dims,
                        expected_dims,
                        cfg.effective_embedding_model,
                    )
                    self._connected = False
                    return False
            self._connected = True
            logger.info(
                "pgvector connected — chunk_embeddings table ready (vector(%d), model %s)",
                expected_dims,
                cfg.effective_embedding_model,
            )
            return True
        except Exception as exc:
            logger.error("pgvector connection failed: %s", exc)
            self._connected = False
            return False

    async def replace_by_source(
        self, source: str, chunks: List[Dict], embeddings: List[List[float]]
    ) -> int:
        """Atomically replace all chunks of a source.

        Deletes every existing row for the source, then inserts the new set —
        in one transaction, so a failure leaves the old version intact. This is
        what makes re-uploading a modified file replace stale chunks instead of
        ON CONFLICT DO NOTHING silently keeping them.
        """
        if not self._engine:
            logger.error("replace_by_source called before connect()")
            return 0
        async with self._engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM chunk_embeddings WHERE source = :src"),
                {"src": source},
            )
            inserted = 0
            for chunk, emb in zip(chunks, embeddings):
                emb_str = "[" + ",".join(str(v) for v in emb) + "]"
                result = await conn.execute(
                    text("""
                        INSERT INTO chunk_embeddings
                            (chunk_id, source, content, chunk_index, embedding, token_count,
                             page_start, page_end, origin_type)
                        VALUES (:cid, :src, :content, :idx, :emb, :tok, :pstart, :pend, :otype)
                        ON CONFLICT (chunk_id) DO NOTHING
                    """),
                    {
                        "cid": chunk["chunk_id"],
                        "src": chunk["source"],
                        "content": chunk["content"],
                        "idx": chunk.get("chunk_index", 0),
                        "emb": emb_str,
                        "tok": chunk.get("token_count", 0),
                        "pstart": chunk.get("page_start"),
                        "pend": chunk.get("page_end"),
                        "otype": chunk.get("origin_type"),
                    },
                )
                inserted += result.rowcount
            return inserted

    async def delete_by_source(self, source: str) -> int:
        """Delete all chunk rows for a source. Returns the number of rows removed."""
        if not self._engine:
            logger.error("delete_by_source called before connect()")
            return 0
        async with self._engine.begin() as conn:
            result = await conn.execute(
                text("DELETE FROM chunk_embeddings WHERE source = :src"),
                {"src": source},
            )
            return result.rowcount or 0

    async def search(self, query_embedding: List[float], top_k: int = 20) -> List[Dict]:
        if not self._engine:
            return []
        try:
            emb_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
            async with self._engine.connect() as conn:
                await conn.execute(text("SET ivfflat.probes = 10"))
                rows = await conn.execute(
                    text("""
                        SELECT chunk_id, source, content, chunk_index, token_count,
                               page_start, page_end, origin_type,
                               1 - (embedding <=> :emb) AS vector_score
                        FROM chunk_embeddings
                        ORDER BY embedding <=> :emb
                        LIMIT :k
                    """),
                    {"emb": emb_str, "k": top_k},
                )
                return [dict(r._mapping) for r in rows]
        except Exception as exc:
            logger.error("vector search failed: %s", exc)
            return []

    async def count(self) -> int:
        if not self._engine:
            return 0
        try:
            async with self._engine.connect() as conn:
                result = await conn.execute(text("SELECT COUNT(*) FROM chunk_embeddings"))
                return result.scalar() or 0
        except Exception as exc:
            logger.error("vector count failed: %s", exc)
            return 0

    async def is_connected(self) -> bool:
        if not self._connected or not self._engine:
            return False
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


vector_store = VectorStore()


def get_engine():
    return vector_store._engine
