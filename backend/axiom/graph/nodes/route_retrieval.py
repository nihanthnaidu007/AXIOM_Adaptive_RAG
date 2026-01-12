"""AXIOM Route Retrieval Node - Fully Implemented."""

from datetime import datetime, timezone
from typing import Any, Dict

from axiom.graph.state import PipelineTraceStep


def route_retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure routing node — no LLM call, no external dependency.
    
    Reads classification.retrieval_strategy from state.
    Sets state["retrieval_strategy"] so the conditional edge can route.
    Logs which strategy was chosen and why to trace_steps.
    """
    start_time = datetime.now(timezone.utc)
    
    classification = state.get("classification")
    
    if classification:
        strategy = classification.retrieval_strategy
        query_type = classification.query_type
        reasoning = classification.reasoning
        entities = classification.entities
    else:
        # Fallback if no classification
        strategy = "hybrid"
        query_type = "unknown"
        reasoning = "No classification available, defaulting to hybrid"
        entities = []
    
    state["retrieval_strategy"] = strategy
    
    # Calculate duration
    end_time = datetime.now(timezone.utc)
    duration_ms = (end_time - start_time).total_seconds() * 1000
    
    # Ensure trace_steps exists
    if "trace_steps" not in state or state["trace_steps"] is None:
        state["trace_steps"] = []
    
    state["trace_steps"].append(PipelineTraceStep(
        node_name="route_retrieval",
        status="complete",
        started_at=start_time.isoformat(),
        duration_ms=round(duration_ms, 2),
        summary=f"Routing to {strategy.upper()} retrieval — query type: {query_type}",
        detail={
            "strategy": strategy,
            "query_type": query_type,
            "reasoning": reasoning,
            "entities": entities
        }
    ))
    
    return state
