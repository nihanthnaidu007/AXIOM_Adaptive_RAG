"""AXIOM Cache Check Node - Real Redis semantic cache lookup."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from axiom.graph.state import CacheCheckResult, PipelineTraceStep, RAGASScores
from axiom.cache.semantic_cache import semantic_cache
from axiom.retrieval.embeddings import embed_text
from axiom.config import get_config

logger = logging.getLogger(__name__)


async def check_cache_node(state: Dict[str, Any]) -> Dict[str, Any]:
    start_time = datetime.now(timezone.utc)
    cfg = get_config()
    user_query = state.get("user_query", "")

    if not await semantic_cache.is_connected():
        state["cache_result"] = CacheCheckResult(hit=False, similarity=0.0)
        state["served_from_cache"] = False
        end_time = datetime.now(timezone.utc)
        _append_trace(state, start_time, end_time, "Cache MISS (Redis not connected)", {
            "mode": "fallback", "threshold": cfg.cache_similarity_threshold,
            "similarity": 0.0, "cache_hit": False,
        })
        return state

    try:
        query_embedding = await embed_text(user_query)
    except Exception as exc:
        logger.warning("Embedding failed in check_cache — skipping cache lookup: %s", exc)
        state["cache_result"] = CacheCheckResult(hit=False, similarity=0.0)
        state["served_from_cache"] = False
        end_time = datetime.now(timezone.utc)
        _append_trace(state, start_time, end_time, f"Cache MISS (embedding failed: {str(exc)[:80]})", {
            "mode": "error", "threshold": cfg.cache_similarity_threshold,
            "similarity": 0.0, "cache_hit": False, "error": str(exc),
        })
        return state

    state["query_embedding"] = query_embedding

    result = await semantic_cache.search(user_query, query_embedding, threshold=cfg.cache_similarity_threshold)

    if result is not None:
        cached_answer = result["final_answer"]
        similarity = result["similarity"]
        state["cache_result"] = CacheCheckResult(
            hit=True, similarity=similarity,
            cached_answer=cached_answer, cache_key=result.get("cache_key"),
        )
        state["served_from_cache"] = True
        state["generated_answer"] = cached_answer
        state["final_answer"] = cached_answer
        state["is_complete"] = True
        state["retrieval_strategy"] = result.get("retrieval_strategy", "") or state.get(
            "retrieval_strategy", ""
        )
        state["correction_attempts"] = result.get("correction_attempts", 0)
        faith = result.get("faithfulness_score", 0.0)
        rel = result.get("answer_relevancy", faith)
        ground = result.get("context_groundedness", faith)
        comp = result.get("composite_score", round(faith * 0.5 + rel * 0.3 + ground * 0.2, 4))
        scorer = result.get("scorer_model") or "cached"
        cached_scores = RAGASScores(
            faithfulness=round(faith, 4),
            answer_relevancy=round(rel, 4),
            context_groundedness=round(ground, 4),
            composite_score=round(comp, 4),
            below_threshold=False,
            scorer_model=scorer,
            evaluation_mode="cached",
        )
        state["ragas_scores"] = cached_scores
        state["scores_history"] = [cached_scores]
        state["evaluation_passed"] = True
        state["hallucination_detected"] = False

        end_time = datetime.now(timezone.utc)
        _append_trace(state, start_time, end_time,
            f"CACHE HIT — similarity: {similarity:.3f} — skipping pipeline",
            {"mode": "real", "threshold": cfg.cache_similarity_threshold,
             "similarity": similarity, "cache_hit": True,
             "cache_key": result.get("cache_key")})
    else:
        state["cache_result"] = CacheCheckResult(hit=False, similarity=0.0)
        state["served_from_cache"] = False

        end_time = datetime.now(timezone.utc)
        _append_trace(state, start_time, end_time,
            "CACHE MISS — proceeding to retrieval",
            {"mode": "real", "threshold": cfg.cache_similarity_threshold,
             "similarity": 0.0, "cache_hit": False})

    return state


def _append_trace(state, start, end, summary, detail):
    if "trace_steps" not in state or state["trace_steps"] is None:
        state["trace_steps"] = []
    duration_ms = (end - start).total_seconds() * 1000
    state["trace_steps"].append(PipelineTraceStep(
        node_name="check_cache", status="complete",
        started_at=start.isoformat(), duration_ms=round(duration_ms, 2),
        summary=summary, detail=detail,
    ))
