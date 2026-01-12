"""AXIOM State Schema - The only data contract for the entire graph."""

from typing import TypedDict, Optional, Literal, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone


# --- Sub-models ---

class QueryClassification(BaseModel):
    """Classification result for an incoming query."""
    query_type: Literal["factual", "abstract", "time_sensitive", "multi_hop"]
    retrieval_strategy: Literal["bm25", "vector", "hybrid"]
    reasoning: str
    entities: List[str] = Field(default_factory=list)
    is_multi_hop: bool = False
    sub_queries: List[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """A single retrieved document chunk with scoring metadata."""
    chunk_id: str
    content: str
    source: str
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    pre_rerank_position: Optional[int] = None
    post_rerank_position: Optional[int] = None


class CacheCheckResult(BaseModel):
    """Result of semantic cache lookup."""
    hit: bool = False
    similarity: float = 0.0
    cached_answer: Optional[str] = None
    cached_chunks: Optional[List[RetrievedChunk]] = None
    cache_key: Optional[str] = None


class RAGASScores(BaseModel):
    """RAGAS evaluation scores for a generated answer."""
    faithfulness: Optional[float]  # 0.0-1.0: is answer supported by chunks? None on parse error
    answer_relevancy: Optional[float]  # 0.0-1.0: does answer address the question? None on parse error
    context_groundedness: Optional[float]  # 0.0-1.0: are claims traceable to sources? None on parse error
    composite_score: float  # weighted average
    below_threshold: bool  # True if faithfulness < FAITHFULNESS_THRESHOLD
    scorer_model: str = "mock"  # "ollama/llama3.2" or "mock"
    evaluation_mode: str = "mock"  # "real", "mock", "cached", or "parse_error"


class CorrectionRecord(BaseModel):
    """Record of a query rewrite during the correction loop."""
    iteration: int
    original_query: str
    rewritten_query: str
    rewrite_reasoning: str
    ragas_scores_before: RAGASScores
    retrieval_strategy_used: str
    chunks_retrieved: int


class ConfidenceBand(BaseModel):
    """Confidence assessment of the final answer."""
    label: Literal["VERIFIED", "PROBABLE", "UNCERTAIN", "UNRELIABLE"]
    score: float
    color_token: str  # maps to UI color variable
    reasoning: str


class PipelineTraceStep(BaseModel):
    """A single step in the pipeline execution trace."""
    node_name: str
    status: Literal["pending", "running", "complete", "error", "skipped"]
    started_at: Optional[str] = None  # ISO timestamp
    duration_ms: Optional[float] = None
    summary: str
    detail: Optional[dict] = None


class ParallelRetrievalTiming(BaseModel):
    """Timing metrics for parallel hybrid retrieval."""
    bm25_ms: Optional[float] = None
    vector_ms: Optional[float] = None
    parallel_total_ms: Optional[float] = None
    estimated_sequential_ms: Optional[float] = None
    speedup_factor: Optional[float] = None


# --- Main Graph State ---

class AxiomState(TypedDict, total=False):
    """
    The complete state schema for the AXIOM graph.
    Every node reads from and writes to this state exclusively.
    """
    # Input
    user_query: str
    session_id: str
    
    # Query Classification
    classification: Optional[QueryClassification]
    
    # Cache
    cache_result: Optional[CacheCheckResult]
    served_from_cache: bool
    
    # Retrieval
    active_query: str
    retrieval_strategy: str
    raw_chunks: List[RetrievedChunk]
    reranked_chunks: List[RetrievedChunk]
    parallel_timing: Optional[ParallelRetrievalTiming]
    
    # Decomposition
    decomposed: bool
    sub_query_results: List[dict]
    
    # Generation
    generated_answer: str
    answer_history: List[str]
    
    # Evaluation
    ragas_scores: Optional[RAGASScores]
    evaluation_mode: Optional[str]  # "real", "mock", "cached", or "parse_error"
    scores_history: List[RAGASScores]
    hallucination_detected: Optional[bool]  # None when evaluation ran in mock mode
    evaluation_passed: bool
    
    # Self-Correction
    correction_attempts: int
    correction_history: List[CorrectionRecord]
    
    # Output
    final_answer: str
    confidence: Optional[ConfidenceBand]
    gate_passed: bool
    exhausted_corrections: bool
    is_complete: bool
    
    # Observability
    trace_steps: List[PipelineTraceStep]
    langsmith_trace_id: Optional[str]
    langsmith_trace_url: Optional[str]
    
    # Cached embedding (reused between check_cache and finalize)
    query_embedding: Optional[list]

    # Error
    error: Optional[str]


def create_initial_state(user_query: str, session_id: str) -> AxiomState:
    """Create a properly initialized AxiomState."""
    return AxiomState(
        user_query=user_query,
        session_id=session_id,
        classification=None,
        cache_result=None,
        served_from_cache=False,
        active_query=user_query,
        retrieval_strategy="",
        raw_chunks=[],
        reranked_chunks=[],
        parallel_timing=None,
        decomposed=False,
        sub_query_results=[],
        generated_answer="",
        answer_history=[],
        ragas_scores=None,
        scores_history=[],
        hallucination_detected=False,
        evaluation_passed=False,
        correction_attempts=0,
        correction_history=[],
        final_answer="",
        confidence=None,
        gate_passed=False,
        exhausted_corrections=False,
        is_complete=False,
        trace_steps=[],
        langsmith_trace_id=None,
        langsmith_trace_url=None,
        query_embedding=None,
        error=None
    )
