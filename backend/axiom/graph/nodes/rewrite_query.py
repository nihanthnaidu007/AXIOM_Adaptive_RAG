"""AXIOM Rewrite Query Node - Fully Implemented."""

from datetime import datetime, timezone
from typing import Any, Dict

from axiom.graph.state import CorrectionRecord, PipelineTraceStep
from axiom.llm.client import chat_json

REWRITE_PROMPT = """You are a query optimization specialist for a RAG retrieval system.

A generated answer failed the hallucination detection check.

ORIGINAL USER QUERY: {user_query}
ACTIVE QUERY (used for retrieval): {active_query}
GENERATED ANSWER (failed): {failed_answer}

RAGAS FAILURE ANALYSIS:
- Faithfulness score: {faithfulness:.2f} (threshold: {threshold:.2f}) — answer contained claims not in retrieved chunks
- Answer relevancy: {relevancy:.2f}
- Context groundedness: {groundedness:.2f}

RETRIEVED CHUNKS (what the system found):
{chunk_summary}

PREVIOUS CORRECTION ATTEMPTS: {attempt_count}

TASK: Diagnose WHY retrieval failed and rewrite the query to retrieve better chunks.

Consider: 
- Were the wrong keywords used? (reframe with synonyms or more specific terms)
- Is the query too broad? (add specificity)  
- Are key entities missing from the query? (include them explicitly)
- Would a different angle surface better evidence? (try rephrasing the question)

Respond ONLY with valid JSON:
{{
    "rewrite_reasoning": "2-3 sentences explaining the retrieval failure and what you changed",
    "rewritten_query": "the reformulated query string"
}}"""


async def rewrite_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rewrite the query when hallucination is detected.
    
    This node is fully implemented — only depends on Claude Sonnet.
    The rewrite reasoning is the key signal in the correction loop.
    """
    start_time = datetime.now(timezone.utc)
    
    user_query = state.get("user_query", "")
    active_query = state.get("active_query", user_query)
    generated_answer = state.get("generated_answer", "")
    ragas_scores = state.get("ragas_scores")
    reranked_chunks = state.get("reranked_chunks", [])
    correction_attempts = state.get("correction_attempts", 0)
    retrieval_strategy = state.get("retrieval_strategy", "hybrid")
    
    correction_attempts += 1
    state["correction_attempts"] = correction_attempts
    
    chunk_summaries = []
    for i, chunk in enumerate(reranked_chunks[:3], start=1):
        preview = chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
        chunk_summaries.append(f"Chunk {i}: {preview}")
    chunk_summary = "\n".join(chunk_summaries) if chunk_summaries else "No chunks retrieved"
    
    faithfulness = (ragas_scores.faithfulness if ragas_scores else None) or 0.5
    relevancy = (ragas_scores.answer_relevancy if ragas_scores else None) or 0.6
    groundedness = (ragas_scores.context_groundedness if ragas_scores else None) or 0.5
    
    prompt = REWRITE_PROMPT.format(
        user_query=user_query,
        active_query=active_query,
        failed_answer=generated_answer[:500] + "..." if len(generated_answer) > 500 else generated_answer,
        faithfulness=faithfulness,
        threshold=0.75,
        relevancy=relevancy,
        groundedness=groundedness,
        chunk_summary=chunk_summary,
        attempt_count=correction_attempts
    )
    
    try:
        # Rewrite needs small JSON; reduce max token budget to lower load.
        rewrite_data = await chat_json(prompt, max_tokens=600)
        rewrite_reasoning = rewrite_data.get("rewrite_reasoning", "Query reformulated for better retrieval")
        rewritten_query = rewrite_data.get("rewritten_query", user_query)
    except Exception as e:
        print(f"Rewrite error: {e}")
        rewrite_reasoning = f"Fallback rewrite due to error: {str(e)[:50]}"
        rewritten_query = f"{user_query} specific details evidence"
    
    correction_record = CorrectionRecord(
        iteration=correction_attempts,
        original_query=user_query,
        rewritten_query=rewritten_query,
        rewrite_reasoning=rewrite_reasoning,
        ragas_scores_before=ragas_scores,
        retrieval_strategy_used=retrieval_strategy,
        chunks_retrieved=len(reranked_chunks)
    )
    
    if "correction_history" not in state or state["correction_history"] is None:
        state["correction_history"] = []
    state["correction_history"].append(correction_record)
    
    state["active_query"] = rewritten_query
    
    end_time = datetime.now(timezone.utc)
    duration_ms = (end_time - start_time).total_seconds() * 1000
    
    if "trace_steps" not in state or state["trace_steps"] is None:
        state["trace_steps"] = []
    
    state["trace_steps"].append(PipelineTraceStep(
        node_name="rewrite_query",
        status="complete",
        started_at=start_time.isoformat(),
        duration_ms=round(duration_ms, 2),
        summary=f"Query rewritten (attempt {correction_attempts}/3) — {rewrite_reasoning[:80]}...",
        detail={
            "iteration": correction_attempts,
            "original_query": user_query,
            "rewritten_query": rewritten_query,
            "reasoning": rewrite_reasoning,
            "previous_faithfulness": faithfulness
        }
    ))
    
    return state
