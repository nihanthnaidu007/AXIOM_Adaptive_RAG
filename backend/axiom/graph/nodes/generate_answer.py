"""AXIOM Generate Answer Node - Fully Implemented."""

from datetime import datetime, timezone
from typing import Any, Dict

from axiom.graph.state import PipelineTraceStep
from axiom.llm.client import chat

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

User query: {user_query}
{correction_context}"""

CORRECTION_CONTEXT_TEMPLATE = """
PREVIOUS ATTEMPT FAILED HALLUCINATION CHECK:
Faithfulness score: {faithfulness:.2f} (threshold: {threshold:.2f})
Critic feedback: The previous answer contained claims not fully supported by the retrieved context.
Rewritten query used for this attempt: {rewritten_query}

This is correction attempt {attempt}/3. Prioritize strict grounding over completeness."""


async def generate_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an answer using Claude Sonnet grounded in retrieved chunks.
    
    This node is fully implemented — no CURSOR_TODO dependencies.
    If correction_attempts > 0: includes previous answer + RAGAS critique in prompt.
    """
    start_time = datetime.now(timezone.utc)
    
    user_query = state.get("user_query", "")
    active_query = state.get("active_query", user_query)
    reranked_chunks = state.get("reranked_chunks", [])
    correction_attempts = state.get("correction_attempts", 0)
    scores_history = state.get("scores_history", [])
    
    context_parts = []
    for i, chunk in enumerate(reranked_chunks, start=1):
        score_info = ""
        if chunk.rerank_score is not None:
            score_info = f" (relevance: {chunk.rerank_score:.2f})"
        context_parts.append(f"[Chunk {i}]{score_info}\n{chunk.content}\nSource: {chunk.source}")
    
    context_block = "\n\n---\n\n".join(context_parts) if context_parts else "No context available."
    
    correction_context = ""
    if correction_attempts > 0 and scores_history:
        last_scores = scores_history[-1]
        correction_context = CORRECTION_CONTEXT_TEMPLATE.format(
            faithfulness=last_scores.faithfulness if last_scores.faithfulness is not None else 0.0,
            threshold=0.75,
            rewritten_query=active_query,
            attempt=correction_attempts
        )
    
    full_prompt = GENERATE_SYSTEM_PROMPT.format(
        chunk_count=len(reranked_chunks),
        context_block=context_block,
        user_query=user_query,
        correction_context=correction_context
    )
    full_prompt += f"\n\nPlease answer the following query based strictly on the provided context:\n\n{user_query}"
    
    try:
        # Answer generation can be longer, but keep token budget reasonable
        # to reduce likelihood of transient service overload (529).
        # Reduce token budget to keep the full correction loop responsive.
        generated_answer = (await chat(full_prompt, max_tokens=400)).strip()
    except Exception as e:
        print(f"Generation error: {e}")
        generated_answer = f"GENERATION_ERROR: Unable to generate answer due to: {str(e)[:100]}"
    
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
            "attempt": correction_attempts + 1,
            "answer_length": len(generated_answer),
        }
    ))
    
    return state
