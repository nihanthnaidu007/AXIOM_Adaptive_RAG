"""AXIOM Vector Store - Real pgvector semantic search."""

import logging
from typing import List, Dict, Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import text

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
            self._engine = create_async_engine(self._build_dsn(), pool_size=5, max_overflow=10)
            async with self._engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS chunk_embeddings (
                        id SERIAL PRIMARY KEY,
                        chunk_id TEXT NOT NULL UNIQUE,
                        source TEXT NOT NULL,
                        content TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        embedding vector(1536) NOT NULL,
                        token_count INTEGER,
                        bm25_score FLOAT,
                        ingested_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """))
                row_count = (await conn.execute(text("SELECT COUNT(*) FROM chunk_embeddings"))).scalar() or 0
                lists = max(1, min(100, int(row_count ** 0.5)))
                await conn.execute(text("DROP INDEX IF EXISTS chunk_embeddings_vec_idx"))
                if row_count > 0:
                    await conn.execute(text(
                        f"CREATE INDEX chunk_embeddings_vec_idx "
                        f"ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops) "
                        f"WITH (lists = {lists})"
                    ))
            self._connected = True
            logger.info("pgvector connected — chunk_embeddings table ready")
            return True
        except Exception as exc:
            logger.error("pgvector connection failed: %s", exc)
            self._connected = False
            return False

    async def insert_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> int:
        if not self._engine:
            logger.error("insert_chunks called before connect()")
            return 0
        try:
            inserted = 0
            async with self._engine.begin() as conn:
                for chunk, emb in zip(chunks, embeddings):
                    emb_str = "[" + ",".join(str(v) for v in emb) + "]"
                    result = await conn.execute(
                        text("""
                            INSERT INTO chunk_embeddings
                                (chunk_id, source, content, chunk_index, embedding, token_count)
                            VALUES (:cid, :src, :content, :idx, :emb, :tok)
                            ON CONFLICT (chunk_id) DO NOTHING
                        """),
                        {
                            "cid": chunk["chunk_id"],
                            "src": chunk["source"],
                            "content": chunk["content"],
                            "idx": chunk.get("chunk_index", 0),
                            "emb": emb_str,
                            "tok": chunk.get("token_count", 0),
                        },
                    )
                    inserted += result.rowcount
            return inserted
        except Exception as exc:
            logger.error("insert_chunks failed: %s", exc)
            return 0

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
