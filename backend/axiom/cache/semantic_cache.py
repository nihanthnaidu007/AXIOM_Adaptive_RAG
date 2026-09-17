"""AXIOM Semantic Cache - Redis-backed cosine similarity cache."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import redis.asyncio as aioredis

from axiom.config import get_config

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


class SemanticCache:
    # Index keys are embedding-identity versioned (see _version_tag): a
    # provider/dimension switch must never scan entries written under the
    # old identity — their vectors are incomparable with the new ones.
    _INDEX_PREFIX = "axiom:cache:index"
    _ZINDEX_PREFIX = "axiom:cache:zindex"

    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._connected = False

    async def connect(self) -> bool:
        try:
            cfg = get_config()
            self._redis = aioredis.Redis(
                host=cfg.redis_host,
                port=cfg.redis_port,
                password=cfg.redis_password or None,
                decode_responses=True,
            )
            await self._redis.ping()
            self._connected = True
            return True
        except Exception as exc:
            logger.error("Redis connection failed: %s", exc)
            self._connected = False
            return False

    async def is_connected(self) -> bool:
        if not self._connected or not self._redis:
            return False
        try:
            await self._redis.ping()
            return True
        except Exception:
            return False

    @staticmethod
    def _version_tag() -> str:
        """Embedding identity for key namespacing — model + width.

        Dimension-versioned keys are the standing rule (spec D2): entries
        written under one embedding model/dimension must be unreachable from
        another. Query text hashes alone are not enough — the same question
        re-embedded at a different width would compare incomparable vectors.
        """
        cfg = get_config()
        return f"{cfg.effective_embedding_model}@{cfg.effective_embedding_dimensions}"

    @classmethod
    def _cache_key(cls, user_query: str) -> str:
        h = hashlib.sha256(user_query.encode()).hexdigest()[:12]
        return f"axiom:cache:{cls._version_tag()}:{h}"

    @classmethod
    def _index_key(cls) -> str:
        return f"{cls._INDEX_PREFIX}:{cls._version_tag()}"

    @classmethod
    def _zindex_key(cls) -> str:
        return f"{cls._ZINDEX_PREFIX}:{cls._version_tag()}"

    def _build_cache_entry(self, data: dict, key: str, similarity: float) -> dict:
        """Build a cache result dict from a Redis hash."""
        faith = float(data.get("faithfulness_score", 0) or 0)
        rel = float(data.get("answer_relevancy", faith) or faith)
        ground = float(data.get("context_groundedness", faith) or faith)
        comp_raw = data.get("composite_score")
        composite = (
            float(comp_raw)
            if comp_raw not in (None, "")
            else round(faith * 0.5 + rel * 0.3 + ground * 0.2, 4)
        )
        return {
            "user_query": data.get("user_query", ""),
            "final_answer": data.get("final_answer", ""),
            "retrieval_strategy": data.get("retrieval_strategy", ""),
            "correction_attempts": int(data.get("correction_attempts", 0)),
            "confidence_label": data.get("confidence_label", ""),
            "confidence_score": float(data.get("confidence_score", 0) or 0),
            "faithfulness_score": faith,
            "answer_relevancy": rel,
            "context_groundedness": ground,
            "composite_score": composite,
            "scorer_model": data.get("scorer_model", "cached"),
            "hit_count": int(data.get("hit_count", 0)),
            "created_at": data.get("created_at", ""),
            "similarity": round(similarity, 6),
            "cache_key": key,
        }

    async def search(
        self, user_query: str, query_embedding: list[float], threshold: float = 0.95
    ) -> dict | None:
        if not self._connected or not self._redis:
            return None
        try:
            # Tier 1: exact match on normalised query string
            if user_query:
                exact_key = self._cache_key(user_query.lower().strip())
                if await self._redis.exists(exact_key):
                    data = await self._redis.hgetall(exact_key)
                    if data:
                        entry = self._build_cache_entry(data, exact_key, 1.0)
                        await self._redis.hincrby(exact_key, "hit_count", 1)
                        logger.info("Cache Tier-1 exact hit: %s", exact_key)
                        return entry

            # Tier 2: approximate match over the 200 most recent keys
            keys = await self._redis.zrevrangebyscore(
                self._zindex_key(), "+inf", "-inf", start=0, num=200
            )
            if not keys:
                # Fall back to legacy set index for backward compatibility
                keys = await self._redis.smembers(self._index_key())
            if not keys:
                return None

            async with self._redis.pipeline(transaction=False) as pipe:
                for key in keys:
                    pipe.hget(key, "embedding")
                raw_embeddings = await pipe.execute()

            best_sim = -1.0
            best_key: str | None = None

            for key, raw_emb in zip(keys, raw_embeddings):
                if raw_emb is None:
                    continue
                stored_emb = json.loads(raw_emb)
                # Width guard: a vector of another width is incomparable —
                # cosine against it is meaningless (and versioned keys should
                # already have excluded it; this is defense in depth).
                if len(stored_emb) != len(query_embedding):
                    logger.warning("Skipping cache entry %s with mismatched embedding width", key)
                    continue
                sim = _cosine_similarity(query_embedding, stored_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

            if best_sim >= threshold and best_key:
                data = await self._redis.hgetall(best_key)
                entry = self._build_cache_entry(data, best_key, best_sim)
                await self._redis.hincrby(best_key, "hit_count", 1)
                return entry

            return None
        except Exception as exc:
            logger.error("Cache search failed: %s", exc)
            return None

    async def store(
        self,
        user_query: str,
        query_embedding: list[float],
        state: dict,
    ) -> bool:
        if not self._connected or not self._redis:
            return False
        if not state.get("evaluation_passed", False):
            return False
        try:
            key = self._cache_key(user_query.lower().strip())
            confidence = state.get("confidence")
            ragas = state.get("ragas_scores")

            entry = {
                "user_query": user_query,
                "embedding": json.dumps(query_embedding),
                "final_answer": state.get("final_answer", ""),
                "retrieval_strategy": state.get("retrieval_strategy", ""),
                "correction_attempts": str(state.get("correction_attempts", 0)),
                "confidence_label": getattr(confidence, "label", "") if confidence else "",
                "confidence_score": str(getattr(confidence, "score", 0) if confidence else 0),
                "faithfulness_score": str(getattr(ragas, "faithfulness", 0) if ragas else 0),
                "answer_relevancy": str(getattr(ragas, "answer_relevancy", 0) if ragas else 0),
                "context_groundedness": str(
                    getattr(ragas, "context_groundedness", 0) if ragas else 0
                ),
                "composite_score": str(getattr(ragas, "composite_score", 0) if ragas else 0),
                "scorer_model": str(
                    getattr(ragas, "scorer_model", "unknown") if ragas else "unknown"
                ),
                "hit_count": "0",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            await self._redis.hset(key, mapping=entry)
            await self._redis.zadd(
                self._zindex_key(), {key: datetime.now(timezone.utc).timestamp()}
            )
            await self._redis.expire(key, 604800)
            return True
        except Exception as exc:
            logger.error("Cache store failed: %s", exc)
            return False

    async def clear(self) -> int:
        """Drop all cached entries and both index structures.

        Called whenever the document corpus changes (ingest/delete): cache
        entries carry no source lineage, so a full clear is the only sound
        invalidation — stale answers must never be served against new content.
        """
        if not self._connected or not self._redis:
            return 0
        try:
            deleted = 0
            cursor: int | str = 0
            while True:
                cursor, batch = await self._redis.scan(
                    cursor=cursor, match="axiom:cache:*", count=200
                )
                # The index keys match the glob but are metadata, not entries.
                data_keys = [k for k in batch if k not in (self._zindex_key(), self._index_key())]
                if data_keys:
                    deleted += await self._redis.delete(*data_keys)
                if cursor == 0:
                    break
            await self._redis.delete(self._zindex_key(), self._index_key())
            return deleted
        except Exception as exc:
            logger.error("Cache clear failed: %s", exc)
            return 0

    async def stats(self) -> dict:
        if not self._connected or not self._redis:
            return {"total_entries": 0, "total_hits": 0}
        try:
            count = await self._redis.zcard(self._zindex_key())
            keys = await self._redis.zrange(self._zindex_key(), 0, -1)
            # Batch all hit_count reads into a single pipeline round-trip
            async with self._redis.pipeline(transaction=False) as pipe:
                for key in keys:
                    pipe.hget(key, "hit_count")
                hit_counts = await pipe.execute()
            total_hits = sum(int(hc or 0) for hc in hit_counts)
            return {"total_entries": count, "total_hits": total_hits}
        except Exception as exc:
            logger.error("Cache stats failed: %s", exc)
            return {"total_entries": 0, "total_hits": 0}


semantic_cache = SemanticCache()
