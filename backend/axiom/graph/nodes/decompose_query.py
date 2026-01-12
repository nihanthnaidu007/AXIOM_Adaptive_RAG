"""AXIOM Decompose Query Node — Multi-hop query decomposition with parallel sub-query execution."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from axiom.graph.state import PipelineTraceStep, RetrievedChunk
from axiom.graph.sub_query_runner import run_sub_queries_parallel, synthesize_sub_answers

logger = logging.getLogger(__name__)


async def decompose_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    start_time = datetime.now(timezone.utc)
    classification = state.get("classification")

    if "trace_steps" not in state or state["trace_steps"] is None:
        state["trace_steps"] = []
    if "answer_history" not in state or state["answer_history"] is None:
        state["answer_history"] = []

    should_skip = (
        not classification
        or not classification.is_multi_hop
        or not classification.sub_queries
        or len(classification.sub_queries) < 2
        or state.get("correction_attempts", 0) > 0
    )

    if should_skip:
        state["decomposed"] = False
        state["sub_query_results"] = []

        end_time = datetime.now(timezone.utc)
        duration_ms = (end_time - start_time).total_seconds() * 1000
        state["trace_steps"].append(PipelineTraceStep(
            node_name="decompose_query",
            status="skipped",
            started_at=start_time.isoformat(),
            duration_ms=round(duration_ms, 2),
            summary="Not multi-hop or correction iteration — skipping decomposition",
            detail={
                "is_multi_hop": classification.is_multi_hop if classification else False,
                "correction_attempts": state.get("correction_attempts", 0),
            },
        ))
        return state

    sub_queries = classification.sub_queries
    strategy = classification.retrieval_strategy

    sub_results = await run_sub_queries_parallel(
        sub_queries=sub_queries,
        parent_session_id=state["session_id"],
        strategy=strategy,
    )

    synthesized = await synthesize_sub_answers(
        original_query=state["user_query"],
        sub_query_results=sub_results,
    )

    state["decomposed"] = True
    state["sub_query_results"] = list(sub_results)
    state["generated_answer"] = synthesized
    state["answer_history"].append(synthesized)

    all_chunks = []
    seen_ids: set[str] = set()
    for result in sub_results:
        for chunk in result.get("chunks_used", []):
            cid = chunk.get("chunk_id", "")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(
                    RetrievedChunk(
                        chunk_id=cid,
                        content=chunk.get("content", ""),
                        source=chunk.get("source", "unknown"),
                        rerank_score=chunk.get("rerank_score"),
                    )
                )

    if all_chunks:
        state["raw_chunks"] = all_chunks

    end_time = datetime.now(timezone.utc)
    duration_ms = (end_time - start_time).total_seconds() * 1000
    state["trace_steps"].append(PipelineTraceStep(
        node_name="decompose_query",
        status="complete",
        started_at=start_time.isoformat(),
        duration_ms=round(duration_ms, 2),
        summary=f"Decomposed into {len(sub_queries)} sub-queries — synthesized answer ready",
        detail={
            "sub_queries": sub_queries,
            "sub_query_count": len(sub_queries),
            "successful_sub_queries": sum(1 for r in sub_results if r.get("success")),
            "strategy": strategy,
            "synthesized_length": len(synthesized),
        },
    ))
    return state
