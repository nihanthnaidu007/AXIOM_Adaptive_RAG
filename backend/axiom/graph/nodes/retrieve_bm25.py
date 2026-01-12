"""AXIOM BM25 Retrieval Node - Real BM25 keyword search."""

from datetime import datetime, timezone
from typing import Any, Dict
from axiom.graph.state import RetrievedChunk, PipelineTraceStep
from axiom.retrieval.bm25_index import bm25_index
from axiom.config import get_config


async def retrieve_bm25_node(state: Dict[str, Any]) -> Dict[str, Any]:
    start_time = datetime.now(timezone.utc)
    active_query = state.get("active_query", state.get("user_query", ""))

    if "trace_steps" not in state or state["trace_steps"] is None:
        state["trace_steps"] = []

    if bm25_index.is_empty():
        state["raw_chunks"] = []
        state["trace_steps"].append(PipelineTraceStep(
            node_name="retrieve_bm25", status="complete",
            started_at=start_time.isoformat(), duration_ms=0,
            summary="BM25 index empty — no chunks returned",
            detail={"index_empty": True, "chunk_count": 0}
        ))
        return state

    results = bm25_index.search(active_query, top_k=get_config().bm25_top_k)
    chunks = []
    for i, r in enumerate(results):
        chunks.append(RetrievedChunk(
            chunk_id=r.get("chunk_id", f"bm25_{i}"),
            content=r.get("content", ""),
            source=r.get("source", "unknown"),
            bm25_score=float(r.get("bm25_score", 0.0)),
            vector_score=None,
            rrf_score=None,
            rerank_score=None,
            pre_rerank_position=i,
            post_rerank_position=None
        ))

    state["raw_chunks"] = chunks
    end_time = datetime.now(timezone.utc)
    duration_ms = (end_time - start_time).total_seconds() * 1000
    top_score = chunks[0].bm25_score if chunks else 0.0

    state["trace_steps"].append(PipelineTraceStep(
        node_name="retrieve_bm25", status="complete",
        started_at=start_time.isoformat(), duration_ms=round(duration_ms, 2),
        summary=f"BM25 retrieval: {len(chunks)} chunks, top score: {top_score:.3f}",
        detail={"index_empty": False, "chunk_count": len(chunks), "top_score": top_score, "query": active_query[:60]}
    ))
    return state
