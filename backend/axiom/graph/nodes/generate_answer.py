"""AXIOM Generate Answer Node - Fully Implemented."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from axiom.graph.state import PipelineTraceStep
from axiom.llm.client import chat
from axiom.config import get_config

logger = logging.getLogger(__name__)

GENERATE_SYSTEM_PROMPT = """You are AXIOM, a document intelligence system that generates 
precise, grounded answers from retrieved document chunks.

RETRIEVED CONTEXT (top {chunk_count} chunks, ranked by relevance):
{context_block}

STRICT RULES:
1. Answer ONLY from the provided context. Do not use external knowledge.
2. If the context does not contain sufficient information, say exactly:
   "INSUFFICIENT_CONTEXT: [brief explanation of what is missing]"
3. Every factual claim must be traceable to at least one chunk.
4. Be concise and direct. Start with the answer, then support it.
5. Do not cite chunk IDs or mention retrieval mechanics to the user.

User query: <user_query>{user_query}</user_query>
{correction_context}"""

CORRECTION_CONTEXT_TEMPLATE = """
PREVIOUS ATTEMPT FAILED HALLUCINATION CHECK:
Faithfulness score: {faithfulness:.2f} (threshold: {threshold:.2f})
Critic feedback: The previous answer contained claims not fully supported by the retrieved context.
Rewritten query used for this attempt: {rewritten_query}

This is correction attempt {attempt}/3. Prioritize strict grounding over completeness."""

WEB_AUGMENTED_PROMPT = """You are AXIOM, a document intelligence system.

RETRIEVED DOCUMENT CONTEXT ({doc_chunk_count} chunks from indexed documents):
{doc_context_block}

WEB SEARCH CONTEXT ({web_chunk_count} results from live web search):
{web_context_block}

STRICT RULES:
1. Answer using BOTH document and web context where relevant.
2. Prefer document context for factual claims about the ingested corpus.
3. Prefer web context for current information not present in documents.
4. If neither source contains sufficient information, say exactly:
   "INSUFFICIENT_CONTEXT: [brief explanation of what is missing]"
5. Every factual claim must be traceable to at least one source.
6. Do not cite chunk IDs or mention retrieval mechanics to the user.

User query: {user_query}
{correction_context}"""


async def generate_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an answer using Claude Sonnet grounded in retrieved chunks.

    If correction_attempts > 0: includes previous answer + RAGAS critique in prompt.
    """
    start_time = datetime.now(timezone.utc)
    
    user_query = state.get("user_query", "")
    active_query = state.get("active_query", user_query)
    reranked_chunks = state.get("reranked_chunks", [])
    web_search_chunks = state.get("web_search_chunks", [])
    correction_attempts = state.get("correction_attempts", 0)
    scores_history = state.get("scores_history", [])

    correction_context = ""
    if correction_attempts > 0 and scores_history:
        last_scores = scores_history[-1]
        correction_context = CORRECTION_CONTEXT_TEMPLATE.format(
            faithfulness=last_scores.faithfulness if last_scores.faithfulness is not None else 0.0,
            threshold=get_config().faithfulness_threshold,
            rewritten_query=active_query,
            attempt=correction_attempts
        )

    # Build document context block from reranked corpus chunks
    doc_context_parts = []
    for i, chunk in enumerate(reranked_chunks, start=1):
        score_info = ""
        if chunk.rerank_score is not None:
            score_info = f" (relevance: {chunk.rerank_score:.2f})"
        doc_context_parts.append(
            f"[Doc {i}]{score_info}\n{chunk.content}\nSource: {chunk.source}"
        )

    # Build web context block from Tavily results
    web_context_parts = []
    for i, w in enumerate(web_search_chunks, start=1):
        score_info = f" (score: {w.get('score', 0.0):.2f})" if w.get("score") else ""
        title = w.get("title", "")
        title_line = f" — {title}" if title else ""
        web_context_parts.append(
            f"[Web {i}]{score_info}{title_line}\n{w.get('content', '')}\nSource: {w.get('url', '')}"
        )

    doc_count = len(doc_context_parts)
    web_count = len(web_context_parts)

    # Record chunk counts in state for the API response
    state["document_chunk_count"] = doc_count
    state["web_chunk_count"] = web_count

    if web_context_parts:
        # Web-augmented path: use the provenance-aware prompt
        doc_block = "\n\n---\n\n".join(doc_context_parts) if doc_context_parts else "No document context available."
        web_block = "\n\n---\n\n".join(web_context_parts)
        full_prompt = WEB_AUGMENTED_PROMPT.format(
            doc_chunk_count=doc_count,
            doc_context_block=doc_block,
            web_chunk_count=web_count,
            web_context_block=web_block,
            user_query=user_query,
            correction_context=correction_context,
        )
        full_prompt += f"\n\nPlease answer the following query based on the provided context:\n\n{user_query}"
    else:
        # Document-only path: existing prompt, no change in behavior
        context_block = "\n\n---\n\n".join(doc_context_parts) if doc_context_parts else "No context available."
        full_prompt = GENERATE_SYSTEM_PROMPT.format(
            chunk_count=doc_count,
            context_block=context_block,
            user_query=user_query,
            correction_context=correction_context,
        )
        full_prompt += f"\n\nPlease answer the following query based strictly on the provided context:\n\n{user_query}"
    
    try:
        # Answer generation can be longer, but keep token budget reasonable
        # to reduce likelihood of transient service overload (529).
        # Reduce token budget to keep the full correction loop responsive.
        generated_answer = (await chat(full_prompt, max_tokens=1000)).strip()
    except Exception as e:
        logger.warning("Generation error: %s", e)
        generated_answer = "Answer generation failed. Please try again."
    
    state["generated_answer"] = generated_answer
    
    if "answer_history" not in state or state["answer_history"] is None:
        state["answer_history"] = []
    state["answer_history"].append(generated_answer)
    
    end_time = datetime.now(timezone.utc)
    duration_ms = (end_time - start_time).total_seconds() * 1000
    
    if "trace_steps" not in state or state["trace_steps"] is None:
        state["trace_steps"] = []
    
    state["trace_steps"].append(PipelineTraceStep(
        node_name="generate_answer",
        status="complete",
        started_at=start_time.isoformat(),
        duration_ms=round(duration_ms, 2),
        summary=f"Generated answer using {len(reranked_chunks)} chunks (attempt {correction_attempts + 1})",
        detail={
            "chunk_count": len(reranked_chunks),
            "web_chunk_count": web_count,
            "document_chunk_count": doc_count,
            "web_augmented": bool(web_context_parts),
            "attempt": correction_attempts + 1,
            "answer_length": len(generated_answer),
        }
    ))
    
    return state
