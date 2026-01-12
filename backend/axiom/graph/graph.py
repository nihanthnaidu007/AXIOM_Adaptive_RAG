"""AXIOM Graph Compilation - LangGraph StateGraph with cyclic support."""

from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from axiom.graph.state import AxiomState
from axiom.config import get_config
from axiom.graph.nodes.classify_query import classify_query_node
from axiom.graph.nodes.check_cache import check_cache_node
from axiom.graph.nodes.route_retrieval import route_retrieval_node
from axiom.graph.nodes.retrieve_bm25 import retrieve_bm25_node
from axiom.graph.nodes.retrieve_vector import retrieve_vector_node
from axiom.graph.nodes.retrieve_hybrid import retrieve_hybrid_node
from axiom.graph.nodes.decompose_query import decompose_query_node
from axiom.graph.nodes.rerank_chunks import rerank_chunks_node
from axiom.graph.nodes.generate_answer import generate_answer_node
from axiom.graph.nodes.evaluate_answer import evaluate_answer_node
from axiom.graph.nodes.rewrite_query import rewrite_query_node
from axiom.graph.nodes.finalize_answer import finalize_answer_node


def _route_from_cache(state: Dict[str, Any]) -> str:
    """Route based on cache hit/miss."""
    if state.get("served_from_cache", False):
        return "finalize_answer"
    return "route_retrieval"


def _route_retrieval_strategy(state: Dict[str, Any]) -> str:
    """Route to appropriate retrieval node based on strategy."""
    strategy = state.get("retrieval_strategy", "hybrid")
    if strategy == "bm25":
        return "retrieve_bm25"
    elif strategy == "vector":
        return "retrieve_vector"
    else:
        return "retrieve_hybrid"


def _route_from_rerank(state: Dict[str, Any]) -> str:
    """Skip generate_answer when decomposition already produced a synthesized answer."""
    if state.get("decomposed", False):
        return "evaluate_answer"
    return "generate_answer"


def _route_evaluation(state: Dict[str, Any]) -> str:
    """Route based on evaluation result."""
    if state.get("evaluation_passed", False):
        return "finalize_answer"
    elif state.get("correction_attempts", 0) < get_config().max_correction_attempts:
        return "rewrite_query"
    else:
        return "finalize_answer"


def build_graph():
    """
    Build and compile the AXIOM StateGraph.
    
    Graph Flow:
    1. classify_query → check_cache
    2. check_cache → finalize_answer (hit) | route_retrieval (miss)
    3. route_retrieval → retrieve_bm25 | retrieve_vector | retrieve_hybrid
    4. retrieve_* → decompose_query → rerank_chunks → generate_answer → evaluate_answer
    5. evaluate_answer → finalize_answer (pass) | rewrite_query (fail, attempts < 3) | finalize_answer (exhausted)
    6. rewrite_query → route_retrieval (re-enter retrieval)
    7. finalize_answer → END
    """
    workflow = StateGraph(AxiomState)
    
    # Register all nodes
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("route_retrieval", route_retrieval_node)
    workflow.add_node("retrieve_bm25", retrieve_bm25_node)
    workflow.add_node("retrieve_vector", retrieve_vector_node)
    workflow.add_node("retrieve_hybrid", retrieve_hybrid_node)
    workflow.add_node("decompose_query", decompose_query_node)
    workflow.add_node("rerank_chunks", rerank_chunks_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("evaluate_answer", evaluate_answer_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("finalize_answer", finalize_answer_node)
    
    # Entry point
    workflow.set_entry_point("classify_query")
    
    # classify_query → check_cache (always)
    workflow.add_edge("classify_query", "check_cache")
    
    # check_cache → finalize_answer (hit) | route_retrieval (miss)
    workflow.add_conditional_edges(
        "check_cache",
        _route_from_cache,
        {
            "finalize_answer": "finalize_answer",
            "route_retrieval": "route_retrieval"
        }
    )
    
    # route_retrieval → one of three retrieval nodes
    workflow.add_conditional_edges(
        "route_retrieval",
        _route_retrieval_strategy,
        {
            "retrieve_bm25": "retrieve_bm25",
            "retrieve_vector": "retrieve_vector",
            "retrieve_hybrid": "retrieve_hybrid"
        }
    )
    
    # All three retrieval paths → decompose_query
    workflow.add_edge("retrieve_bm25", "decompose_query")
    workflow.add_edge("retrieve_vector", "decompose_query")
    workflow.add_edge("retrieve_hybrid", "decompose_query")
    
    # decompose_query → rerank_chunks (always)
    workflow.add_edge("decompose_query", "rerank_chunks")
    
    # rerank_chunks → generate_answer (normal) | evaluate_answer (decomposed, answer already synthesized)
    workflow.add_conditional_edges(
        "rerank_chunks",
        _route_from_rerank,
        {
            "evaluate_answer": "evaluate_answer",
            "generate_answer": "generate_answer",
        }
    )
    
    # generate_answer → evaluate_answer
    workflow.add_edge("generate_answer", "evaluate_answer")
    
    # evaluate_answer: passed → finalize | failed + attempts < 3 → rewrite | failed + exhausted → finalize
    workflow.add_conditional_edges(
        "evaluate_answer",
        _route_evaluation,
        {
            "finalize_answer": "finalize_answer",
            "rewrite_query": "rewrite_query"
        }
    )
    
    # rewrite_query → route_retrieval (re-enter retrieval with new query)
    workflow.add_edge("rewrite_query", "route_retrieval")
    
    # finalize_answer → END
    workflow.add_edge("finalize_answer", END)
    
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# Global compiled graph instance
_compiled_graph = None


def get_graph():
    """Get or create the compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def get_graph_node_names():
    """Get list of all node names in the graph."""
    return [
        "classify_query",
        "check_cache",
        "route_retrieval",
        "retrieve_bm25",
        "retrieve_vector",
        "retrieve_hybrid",
        "decompose_query",
        "rerank_chunks",
        "generate_answer",
        "evaluate_answer",
        "rewrite_query",
        "finalize_answer"
    ]
