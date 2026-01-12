"""AXIOM Semantic Cache - Redis-backed cosine similarity cache."""

import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis

from axiom.config import get_config

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    _INDEX_KEY = "axiom:cache:index"
    _ZINDEX_KEY = "cache_index"

    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._connected = False

    async def connect(self) -> bool:
        try:
            cfg = get_config()
            self._redis = aioredis.Redis(
                host=cfg.redis_host, port=cfg.redis_port, decode_responses=True
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
    def _cache_key(user_query: str) -> str:
        h = hashlib.sha256(user_query.encode()).hexdigest()[:12]
        return f"axiom:cache:{h}"

    def _build_cache_entry(self, data: dict, key: str, similarity: float) -> dict:
        """Build a cache result dict from a Redis hash."""
        faith = float(data.get("faithfulness_score", 0) or 0)
        rel = float(data.get("answer_relevancy", faith) or faith)
        ground = float(data.get("context_groundedness", faith) or faith)
        comp_raw = data.get("composite_score")
        composite = float(comp_raw) if comp_raw not in (None, "") else round(
            faith * 0.5 + rel * 0.3 + ground * 0.2, 4
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
                self._ZINDEX_KEY, "+inf", "-inf", start=0, num=200
            )
            if not keys:
                # Fall back to legacy set index for backward compatibility
                keys = await self._redis.smembers(self._INDEX_KEY)
            if not keys:
                return None

            best_sim = -1.0
            best_key: str | None = None

            for key in keys:
                raw_emb = await self._redis.hget(key, "embedding")
                if not raw_emb:
                    continue
                stored_emb = json.loads(raw_emb)
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
                "faithfulness_score": str(
                    getattr(ragas, "faithfulness", 0) if ragas else 0
                ),
                "answer_relevancy": str(
                    getattr(ragas, "answer_relevancy", 0) if ragas else 0
                ),
                "context_groundedness": str(
                    getattr(ragas, "context_groundedness", 0) if ragas else 0
                ),
                "composite_score": str(
                    getattr(ragas, "composite_score", 0) if ragas else 0
                ),
                "scorer_model": str(
                    getattr(ragas, "scorer_model", "ollama/llama3.2") if ragas else "unknown"
                ),
                "hit_count": "0",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            await self._redis.hset(key, mapping=entry)
            await self._redis.sadd(self._INDEX_KEY, key)
            await self._redis.zadd(self._ZINDEX_KEY, {key: datetime.now(timezone.utc).timestamp()})
            await self._redis.expire(key, 604800)
            return True
        except Exception as exc:
            logger.error("Cache store failed: %s", exc)
            return False

    async def stats(self) -> dict:
        if not self._connected or not self._redis:
            return {"total_entries": 0, "total_hits": 0}
        try:
            keys = await self._redis.smembers(self._INDEX_KEY)
            total_hits = 0
            for key in keys:
                hc = await self._redis.hget(key, "hit_count")
                total_hits += int(hc or 0)
            return {"total_entries": len(keys), "total_hits": total_hits}
        except Exception as exc:
            logger.error("Cache stats failed: %s", exc)
            return {"total_entries": 0, "total_hits": 0}


semantic_cache = SemanticCache()
