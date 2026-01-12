"""AXIOM Hybrid Retrieval Node - Real async parallel BM25 + pgvector with RRF fusion."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict
from axiom.graph.state import RetrievedChunk, PipelineTraceStep, ParallelRetrievalTiming
from axiom.retrieval.bm25_index import bm25_index
from axiom.retrieval.vector_store import vector_store
from axiom.retrieval.embeddings import embed_text
from axiom.retrieval.hybrid_fusion import reciprocal_rank_fusion


async def retrieve_hybrid_node(state: Dict[str, Any]) -> Dict[str, Any]:
    start_time = datetime.now(timezone.utc)
    active_query = state.get("active_query", state.get("user_query", ""))

    if "trace_steps" not in state or state["trace_steps"] is None:
        state["trace_steps"] = []

    bm25_empty = bm25_index.is_empty()
    try:
        vector_count = await vector_store.count()
        vector_empty = vector_count == 0
    except Exception:
        vector_empty = True

    if bm25_empty and vector_empty:
        state["raw_chunks"] = []
        state["parallel_timing"] = ParallelRetrievalTiming(
            bm25_ms=None, vector_ms=None,
            parallel_total_ms=None, estimated_sequential_ms=None, speedup_factor=None
        )
        state["trace_steps"].append(PipelineTraceStep(
            node_name="retrieve_hybrid", status="complete",
            started_at=start_time.isoformat(), duration_ms=0,
            summary="Both indexes empty — no chunks returned",
            detail={"both_empty": True}
        ))
        return state

    try:
        embedding = await embed_text(active_query)
    except Exception:
        embedding = None

    wall_start = time.monotonic()

    async def run_bm25():
        t = time.monotonic()
        if bm25_empty:
            return [], 0.0
        results = bm25_index.search(active_query, top_k=20)
        return results, (time.monotonic() - t) * 1000

    async def run_vector():
        t = time.monotonic()
        if vector_empty or embedding is None:
            return [], 0.0
        try:
            results = await vector_store.search(embedding, top_k=20)
        except Exception:
            results = []
        return results, (time.monotonic() - t) * 1000

    (bm25_results, bm25_ms), (vector_results, vector_ms) = await asyncio.gather(
        run_bm25(), run_vector()
    )

    parallel_total_ms = (time.monotonic() - wall_start) * 1000
    estimated_sequential_ms = bm25_ms + vector_ms
    speedup_factor = round(estimated_sequential_ms / parallel_total_ms, 2) if parallel_total_ms > 0 else 1.0

    def to_retrieved_chunks(results, strategy):
        out = []
        for i, r in enumerate(results):
            out.append(RetrievedChunk(
                chunk_id=r.get("chunk_id", f"{strategy}_{i}"),
                content=r.get("content", ""),
                source=r.get("source", "unknown"),
                bm25_score=float(r.get("bm25_score", 0.0)) if strategy == "bm25" else None,
                vector_score=float(r.get("vector_score", 0.0)) if strategy == "vector" else None,
                rrf_score=None,
                rerank_score=None,
                pre_rerank_position=i,
                post_rerank_position=None
            ))
        return out

    bm25_chunks = to_retrieved_chunks(bm25_results, "bm25")
    vector_chunks = to_retrieved_chunks(vector_results, "vector")
    fused = reciprocal_rank_fusion(bm25_chunks, vector_chunks)

    state["raw_chunks"] = fused
    state["parallel_timing"] = ParallelRetrievalTiming(
        bm25_ms=round(bm25_ms, 2),
        vector_ms=round(vector_ms, 2),
        parallel_total_ms=round(parallel_total_ms, 2),
        estimated_sequential_ms=round(estimated_sequential_ms, 2),
        speedup_factor=speedup_factor
    )

    end_time = datetime.now(timezone.utc)
    duration_ms = (end_time - start_time).total_seconds() * 1000

    state["trace_steps"].append(PipelineTraceStep(
        node_name="retrieve_hybrid", status="complete",
        started_at=start_time.isoformat(), duration_ms=round(duration_ms, 2),
        summary=f"Hybrid retrieval: {len(fused)} chunks fused, speedup {speedup_factor:.2f}x",
        detail={
            "chunk_count": len(fused),
            "bm25_count": len(bm25_chunks),
            "vector_count": len(vector_chunks),
            "bm25_ms": round(bm25_ms, 2),
            "vector_ms": round(vector_ms, 2),
            "parallel_total_ms": round(parallel_total_ms, 2),
            "speedup_factor": speedup_factor
        }
    ))
    return state
