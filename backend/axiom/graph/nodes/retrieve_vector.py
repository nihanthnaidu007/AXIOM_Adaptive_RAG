"""AXIOM Vector Retrieval Node - Real pgvector semantic search."""

from datetime import datetime, timezone
from typing import Any, Dict
from axiom.graph.state import RetrievedChunk, PipelineTraceStep
from axiom.retrieval.vector_store import vector_store
from axiom.retrieval.embeddings import embed_text
from axiom.config import get_config


async def retrieve_vector_node(state: Dict[str, Any]) -> Dict[str, Any]:
    start_time = datetime.now(timezone.utc)
    active_query = state.get("active_query", state.get("user_query", ""))

    if "trace_steps" not in state or state["trace_steps"] is None:
        state["trace_steps"] = []

    try:
        count = await vector_store.count()
    except Exception:
        count = 0

    if count == 0:
        state["raw_chunks"] = []
        state["trace_steps"].append(PipelineTraceStep(
            node_name="retrieve_vector", status="complete",
            started_at=start_time.isoformat(), duration_ms=0,
            summary="Vector index empty — no chunks returned",
            detail={"index_empty": True, "chunk_count": 0}
        ))
        return state

    try:
        embedding = await embed_text(active_query)
        results = await vector_store.search(embedding, top_k=get_config().vector_top_k)
    except Exception as e:
        state["raw_chunks"] = []
        state["trace_steps"].append(PipelineTraceStep(
            node_name="retrieve_vector", status="error",
            started_at=start_time.isoformat(), duration_ms=0,
            summary=f"Vector retrieval failed: {str(e)[:100]}",
            detail={"error": str(e)}
        ))
        return state

    chunks = []
    for i, r in enumerate(results):
        chunks.append(RetrievedChunk(
            chunk_id=r.get("chunk_id", f"vec_{i}"),
            content=r.get("content", ""),
            source=r.get("source", "unknown"),
            bm25_score=None,
            vector_score=float(r.get("vector_score", 0.0)),
            rrf_score=None,
            rerank_score=None,
            pre_rerank_position=i,
            post_rerank_position=None
        ))

    state["raw_chunks"] = chunks
    end_time = datetime.now(timezone.utc)
    duration_ms = (end_time - start_time).total_seconds() * 1000
    top_score = chunks[0].vector_score if chunks else 0.0

    state["trace_steps"].append(PipelineTraceStep(
        node_name="retrieve_vector", status="complete",
        started_at=start_time.isoformat(), duration_ms=round(duration_ms, 2),
        summary=f"Vector retrieval: {len(chunks)} chunks, top score: {top_score:.3f}",
        detail={"index_empty": False, "chunk_count": len(chunks), "top_score": top_score}
    ))
    return state
