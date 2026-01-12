"""AXIOM Query Classifier Node - Fully Implemented."""

from datetime import datetime, timezone
from typing import Any, Dict

from axiom.graph.state import QueryClassification, PipelineTraceStep
from axiom.llm.client import chat_json

CLASSIFY_PROMPT = """You are a retrieval strategy classifier for an adaptive RAG system.

Given a natural language query, determine:
1. query_type: 
   - "factual" → specific facts, definitions, named entities, dates (use BM25 keyword search)
   - "abstract" → conceptual questions, comparisons, explanations (use vector semantic search)
   - "time_sensitive" → recent events, current state, latest versions (use hybrid search)
   - "multi_hop" → compound questions requiring multiple retrieval passes (use hybrid search)
2. retrieval_strategy: "bm25" | "vector" | "hybrid" (derive from query_type)
3. reasoning: one sentence explaining the classification
4. entities: key named entities, technical terms, and filters in the query
5. is_multi_hop: true only if the question contains two or more independent sub-questions
6. sub_queries: if multi_hop, list each independent sub-question. Otherwise empty list.

Query: {user_query}

Respond ONLY with valid JSON matching this schema exactly. No markdown, no preamble.
{{
  "query_type": "factual|abstract|time_sensitive|multi_hop",
  "retrieval_strategy": "bm25|vector|hybrid",
  "reasoning": "string",
  "entities": ["string"],
  "is_multi_hop": false,
  "sub_queries": []
}}"""


async def classify_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify the incoming query using Claude Sonnet.
    
    This node always runs first and has no CURSOR_TODO dependencies.
    It determines the query type and appropriate retrieval strategy.
    """
    start_time = datetime.now(timezone.utc)
    user_query = state.get("user_query", "")
    
    try:
        prompt = CLASSIFY_PROMPT.format(user_query=user_query)
        # Classification needs a small, deterministic JSON output.
        classification_data = await chat_json(prompt, max_tokens=300)
        
        classification = QueryClassification(
            query_type=classification_data.get("query_type", "factual"),
            retrieval_strategy=classification_data.get("retrieval_strategy", "bm25"),
            reasoning=classification_data.get("reasoning", "Default classification"),
            entities=classification_data.get("entities", []),
            is_multi_hop=classification_data.get("is_multi_hop", False),
            sub_queries=classification_data.get("sub_queries", [])
        )
        
    except Exception as e:
        print(f"Classification error: {e}")
        classification = QueryClassification(
            query_type="factual",
            retrieval_strategy="hybrid",
            reasoning=f"Fallback classification due to error: {str(e)[:50]}",
            entities=[],
            is_multi_hop=False,
            sub_queries=[]
        )
    
    end_time = datetime.now(timezone.utc)
    duration_ms = (end_time - start_time).total_seconds() * 1000
    
    state["classification"] = classification
    state["active_query"] = user_query
    
    if "trace_steps" not in state or state["trace_steps"] is None:
        state["trace_steps"] = []
    
    state["trace_steps"].append(PipelineTraceStep(
        node_name="classify_query",
        status="complete",
        started_at=start_time.isoformat(),
        duration_ms=round(duration_ms, 2),
        summary=f"Classified as {classification.query_type.upper()} → {classification.retrieval_strategy.upper()} strategy",
        detail={
            "query_type": classification.query_type,
            "strategy": classification.retrieval_strategy,
            "entities": classification.entities,
            "is_multi_hop": classification.is_multi_hop,
            "reasoning": classification.reasoning
        }
    ))
    
    return state
