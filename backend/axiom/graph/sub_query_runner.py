"""Standalone sub-query execution pipeline.

Runs a single sub-query through retrieve -> rerank -> generate.
Called in parallel for each sub-query in decompose_query_node.
Pure async function chain — does NOT go through the full LangGraph.
"""

import asyncio
import logging
from typing import Dict, Any, List

from axiom.graph.state import RetrievedChunk
from axiom.retrieval.bm25_index import bm25_index
from axiom.retrieval.vector_store import vector_store
from axiom.retrieval.embeddings import embed_text
from axiom.retrieval.reranker import get_reranker
from axiom.retrieval.hybrid_fusion import reciprocal_rank_fusion
from axiom.llm.client import chat

logger = logging.getLogger(__name__)

SUB_QUERY_PROMPT = """Answer this specific question using only the provided context.
Be concise — 2-4 sentences maximum.

Context:
{context}

Question: {sub_query}

If the context doesn't contain enough information, say "INSUFFICIENT: [what's missing]"
Answer:"""

SYNTHESIS_PROMPT = """You are synthesizing answers to sub-questions into a single coherent response.

Original question: {original_query}

Sub-question answers:
{sub_answers_block}

Write a single coherent answer to the original question that:
1. Integrates all the sub-answers naturally
2. Does not repeat information
3. Flows as a unified response, not a list
4. Is 3-6 sentences long

If any sub-answer says "INSUFFICIENT", acknowledge that aspect is unclear but answer what you can.

Answer:"""


async def _retrieve_for_sub_query(
    sub_query: str, strategy: str, top_k: int = 10
) -> List[RetrievedChunk]:
    """Run retrieval for a single sub-query using the specified strategy."""
    if strategy == "bm25":
        raw = bm25_index.search(sub_query, top_k=top_k)
        return [
            RetrievedChunk(
                chunk_id=c.get("chunk_id", f"bm25-{i}"),
                content=c.get("content", ""),
                source=c.get("source", "unknown"),
                bm25_score=1.0 / (i + 1),
            )
            for i, c in enumerate(raw)
        ]

    if strategy == "vector":
        embedding = await embed_text(sub_query)
        raw = await vector_store.search(embedding, top_k=top_k)
        return [
            RetrievedChunk(
                chunk_id=c.get("chunk_id", f"vec-{i}"),
                content=c.get("content", ""),
                source=c.get("source", "unknown"),
                vector_score=c.get("vector_score", 0.0),
            )
            for i, c in enumerate(raw)
        ]

    # hybrid (default)
    bm25_raw = bm25_index.search(sub_query, top_k=top_k)
    bm25_chunks = [
        RetrievedChunk(
            chunk_id=c.get("chunk_id", f"bm25-{i}"),
            content=c.get("content", ""),
            source=c.get("source", "unknown"),
            bm25_score=1.0 / (i + 1),
        )
        for i, c in enumerate(bm25_raw)
    ]

    embedding = await embed_text(sub_query)
    vec_raw = await vector_store.search(embedding, top_k=top_k)
    vec_chunks = [
        RetrievedChunk(
            chunk_id=c.get("chunk_id", f"vec-{i}"),
            content=c.get("content", ""),
            source=c.get("source", "unknown"),
            vector_score=c.get("vector_score", 0.0),
        )
        for i, c in enumerate(vec_raw)
    ]

    return reciprocal_rank_fusion(bm25_chunks, vec_chunks)


async def run_sub_query(
    sub_query: str,
    parent_session_id: str,
    retrieval_strategy: str = "hybrid",
) -> dict:
    """Execute a single sub-query through retrieve -> rerank -> generate."""
    result: Dict[str, Any] = {
        "sub_query": sub_query,
        "answer": "",
        "chunks_used": [],
        "retrieval_strategy": retrieval_strategy,
        "top_chunk_score": 0.0,
        "success": False,
    }

    try:
        raw_chunks = await _retrieve_for_sub_query(sub_query, retrieval_strategy)

        reranker = get_reranker()
        if raw_chunks:
            reranked = reranker.rerank(sub_query, raw_chunks, top_k=3)
        else:
            reranked = []

        result["chunks_used"] = [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "source": c.source,
                "rerank_score": c.rerank_score,
            }
            for c in reranked
        ]
        if reranked:
            result["top_chunk_score"] = reranked[0].rerank_score or 0.0

        context = "\n---\n".join(c.content for c in reranked) if reranked else "(no context retrieved)"
        prompt = SUB_QUERY_PROMPT.format(context=context, sub_query=sub_query)
        answer = await chat(prompt, max_tokens=500)
        result["answer"] = answer.strip()
        result["success"] = True

    except Exception as exc:
        logger.error("Sub-query failed: %s — %s", sub_query[:60], exc)
        result["answer"] = f"INSUFFICIENT: retrieval failed — {exc}"
        result["success"] = False

    return result


async def run_sub_queries_parallel(
    sub_queries: List[str],
    parent_session_id: str,
    strategy: str = "hybrid",
) -> List[dict]:
    """Run all sub-queries concurrently and return results in input order."""
    return await asyncio.gather(
        *[run_sub_query(q, parent_session_id, strategy) for q in sub_queries]
    )


async def synthesize_sub_answers(
    original_query: str,
    sub_query_results: List[dict],
) -> str:
    """Combine multiple sub-query answers into a single coherent response."""
    parts = []
    for i, r in enumerate(sub_query_results, 1):
        parts.append(f"Q{i}: {r['sub_query']}\nA{i}: {r['answer']}")
    sub_answers_block = "\n\n".join(parts)

    try:
        prompt = SYNTHESIS_PROMPT.format(
            original_query=original_query,
            sub_answers_block=sub_answers_block,
        )
        return (await chat(prompt, max_tokens=1000)).strip()
    except Exception as exc:
        logger.error("Synthesis failed, falling back to concatenation: %s", exc)
        return "\n\n".join(r["answer"] for r in sub_query_results)
