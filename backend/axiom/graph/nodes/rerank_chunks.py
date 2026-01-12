"""AXIOM Rerank Chunks Node - Real cross-encoder reranking."""

from datetime import datetime, timezone
from typing import Any, Dict
from axiom.graph.state import PipelineTraceStep
from axiom.retrieval.reranker import reranker
from axiom.config import get_config


async def rerank_chunks_node(state: Dict[str, Any]) -> Dict[str, Any]:
    start_time = datetime.now(timezone.utc)
    raw_chunks = state.get("raw_chunks", [])
    active_query = state.get("active_query", state.get("user_query", ""))

    if "trace_steps" not in state or state["trace_steps"] is None:
        state["trace_steps"] = []

    if not raw_chunks:
        state["reranked_chunks"] = []
        state["trace_steps"].append(PipelineTraceStep(
            node_name="rerank_chunks", status="complete",
            started_at=start_time.isoformat(), duration_ms=0,
            summary="No chunks to rerank",
            detail={"input_count": 0, "output_count": 0}
        ))
        return state

    reranked = reranker.rerank(active_query, raw_chunks, top_k=get_config().rerank_top_k)
    state["reranked_chunks"] = reranked

    end_time = datetime.now(timezone.utc)
    duration_ms = (end_time - start_time).total_seconds() * 1000
    top_score = reranked[0].rerank_score if reranked else 0.0
    reranker_mode = "real" if reranker.is_loaded() else "fallback"

    sources = [c.source for c in reranked if hasattr(c, 'source')]

    state["trace_steps"].append(PipelineTraceStep(
        node_name="rerank_chunks", status="complete",
        started_at=start_time.isoformat(), duration_ms=round(duration_ms, 2),
        summary=f"Reranked {len(raw_chunks)} → top {len(reranked)}, reranker_mode={reranker_mode}",
        detail={
            "input_count": len(raw_chunks),
            "output_count": len(reranked),
            "model_loaded": reranker.is_loaded(),
            "reranker_mode": reranker_mode,
            "top_score": round(top_score, 4),
            "top_sources": sources[:3]
        }
    ))
    return state
