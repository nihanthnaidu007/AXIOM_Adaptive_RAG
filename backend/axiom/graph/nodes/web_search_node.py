"""AXIOM Web Search Node - Tavily fallback when document corpus is insufficient.

Triggered by two conditions (see graph.py routing):
1. Zero chunks after reranking (corpus empty or query produces no results).
2. Evaluation failed and web_search_used is False (post-correction fallback).

Writes web results to state["web_search_chunks"]. Does NOT append to
state["reranked_chunks"]. generate_answer_node reads both fields and builds
a labeled context block that distinguishes document chunks from web chunks.

Module path: backend/axiom/graph/nodes/web_search_node.py
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from axiom.graph.state import PipelineTraceStep
from axiom.search.web_search import tavily_search, is_tavily_configured
from axiom.retrieval.vector_store import vector_store

logger = logging.getLogger(__name__)


async def _corpus_is_empty() -> bool:
    """Return True if the pgvector store has zero indexed chunks.

    Used to determine Tavily depth tier:
    - True  → advanced search (web must substitute for documents entirely)
    - False → basic search (web supplements existing documents)
    """
    try:
        from sqlalchemy import text as sa_text
        async with vector_store._engine.connect() as conn:
            result = await conn.execute(
                sa_text("SELECT COUNT(*) FROM chunk_embeddings")
            )
            count = result.scalar()
            return (count or 0) == 0
    except Exception as exc:
        logger.warning(
            "corpus_is_empty check failed, defaulting to False: %s", exc
        )
        return False


async def web_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Call Tavily and store web results in state.

    Always sets state["web_search_used"] = True, even on Tavily failure,
    so the routing function knows not to trigger web search again this query.

    Depth selection:
    - Corpus empty → "advanced" (web must fully answer the query)
    - Corpus non-empty → "basic" (web supplements partial document retrieval)
    """
    start_time = datetime.now(timezone.utc)

    if "trace_steps" not in state or state["trace_steps"] is None:
        state["trace_steps"] = []

    # Mark web search as used immediately. This prevents infinite loops if
    # Tavily returns thin results and evaluation still fails after this node.
    state["web_search_used"] = True

    if not is_tavily_configured():
        logger.warning(
            "web_search_node triggered but TAVILY_API_KEY is not configured. "
            "Returning empty web chunks. Query will finalize with available context."
        )
        state["web_search_chunks"] = []
        state["trace_steps"].append(PipelineTraceStep(
            node_name="web_search",
            status="skipped",
            started_at=start_time.isoformat(),
            duration_ms=0,
            summary="Web search skipped — TAVILY_API_KEY not configured",
            detail={"configured": False},
        ))
        return state

    query = state.get("active_query", state.get("user_query", ""))

    # Tiered depth: advanced when corpus is empty (web must fully answer),
    # basic when corpus has documents (web supplements).
    corpus_empty = await _corpus_is_empty()
    depth = "advanced" if corpus_empty else "basic"

    logger.info(
        "web_search_node: query='%s...' depth=%s corpus_empty=%s",
        query[:60], depth, corpus_empty,
    )

    results = await tavily_search(query=query, search_depth=depth)

    state["web_search_chunks"] = results

    end_time = datetime.now(timezone.utc)
    duration_ms = (end_time - start_time).total_seconds() * 1000

    state["trace_steps"].append(PipelineTraceStep(
        node_name="web_search",
        status="complete",
        started_at=start_time.isoformat(),
        duration_ms=round(duration_ms, 2),
        summary=(
            f"Tavily returned {len(results)} web results "
            f"(depth={depth}, corpus_empty={corpus_empty})"
        ),
        detail={
            "result_count": len(results),
            "search_depth": depth,
            "corpus_empty": corpus_empty,
            "query_used": query[:120],
            "top_url": results[0]["url"] if results else None,
        },
    ))

    return state
