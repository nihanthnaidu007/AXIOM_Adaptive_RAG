"""AXIOM Finalize Answer Node - Computes confidence, writes to cache."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from axiom.graph.state import PipelineTraceStep, ConfidenceBand
from axiom.evaluation.thresholds import compute_confidence_band
from axiom.cache.semantic_cache import semantic_cache

logger = logging.getLogger(__name__)


async def finalize_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    start_time = datetime.now(timezone.utc)

    served_from_cache = state.get("served_from_cache", False)
    cache_result = state.get("cache_result")
    ragas_scores = state.get("ragas_scores")
    correction_attempts = state.get("correction_attempts", 0)
    generated_answer = state.get("generated_answer", "")

    if served_from_cache and cache_result and cache_result.cached_answer:
        final_answer = cache_result.cached_answer
        cache_similarity = cache_result.similarity
    else:
        final_answer = generated_answer
        cache_similarity = 0.0

    if ragas_scores:
        confidence = compute_confidence_band(
            ragas_scores=ragas_scores,
            correction_attempts=correction_attempts,
            served_from_cache=served_from_cache,
            cache_similarity=cache_similarity,
        )
    else:
        confidence = ConfidenceBand(
            label="UNCERTAIN", score=0.5,
            color_token="--band-uncertain",
            reasoning="No evaluation scores available",
        )

    state["final_answer"] = final_answer
    state["confidence"] = confidence

    evaluation_passed = state.get("evaluation_passed", False)
    correction_attempts = state.get("correction_attempts", 0)
    served_from_cache = state.get("served_from_cache", False)

    state["gate_passed"] = evaluation_passed or served_from_cache
    state["exhausted_corrections"] = (
        not evaluation_passed and
        not served_from_cache and
        correction_attempts >= 3
    )
    state["is_complete"] = True

    if not served_from_cache and state.get("evaluation_passed", False):
        query_embedding = state.get("query_embedding")
        if query_embedding is None:
            from axiom.retrieval.embeddings import embed_text
            query_embedding = await embed_text(state["user_query"])
        try:
            await semantic_cache.store(
                user_query=state["user_query"],
                query_embedding=query_embedding,
                state=state,
            )
        except Exception as exc:
            logger.warning("Cache write failed in finalize: %s", exc)

    end_time = datetime.now(timezone.utc)
    duration_ms = (end_time - start_time).total_seconds() * 1000

    if "trace_steps" not in state or state["trace_steps"] is None:
        state["trace_steps"] = []

    steps = state.get("trace_steps", [])
    total_duration_ms = sum(step.duration_ms or 0 for step in steps)

    summary_parts = [f"Finalized with {confidence.label} confidence ({confidence.score:.2f})"]
    if correction_attempts > 0:
        summary_parts.append(f"after {correction_attempts} correction(s)")
    if served_from_cache:
        summary_parts.append("(served from cache)")

    state["trace_steps"].append(PipelineTraceStep(
        node_name="finalize_answer", status="complete",
        started_at=start_time.isoformat(), duration_ms=round(duration_ms, 2),
        summary=" ".join(summary_parts),
        detail={
            "confidence_label": confidence.label,
            "confidence_score": confidence.score,
            "color_token": confidence.color_token,
            "reasoning": confidence.reasoning,
            "correction_attempts": correction_attempts,
            "served_from_cache": served_from_cache,
            "total_pipeline_ms": round(total_duration_ms, 2),
            "answer_length": len(final_answer),
        },
    ))
    return state
