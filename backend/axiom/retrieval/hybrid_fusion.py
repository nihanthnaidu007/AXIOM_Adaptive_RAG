"""AXIOM Hybrid Fusion - Reciprocal Rank Fusion Implementation (Fully Implemented)."""

from typing import List, Dict
from axiom.graph.state import RetrievedChunk


def reciprocal_rank_fusion(
    bm25_chunks: List[RetrievedChunk],
    vector_chunks: List[RetrievedChunk],
    k: int = 60
) -> List[RetrievedChunk]:
    """
    Reciprocal Rank Fusion - merge two ranked lists into one.
    
    RRF score for a document d: sum(1 / (k + rank_in_list_i)) where k=60
    
    This is a pure-Python algorithm with no external dependencies.
    
    Args:
        bm25_chunks: Ranked list from BM25 keyword search
        vector_chunks: Ranked list from vector similarity search  
        k: Smoothing constant (default 60, from original RRF paper)
        
    Returns:
        Merged list sorted by RRF score, with rrf_score populated on each chunk
    """
    # Dictionary to accumulate RRF scores by chunk_id
    rrf_scores: Dict[str, float] = {}
    # Dictionary to store chunk objects by chunk_id
    chunks_by_id: Dict[str, RetrievedChunk] = {}
    
    # Process BM25 results
    for rank, chunk in enumerate(bm25_chunks, start=1):
        rrf_score = 1.0 / (k + rank)
        chunk_id = chunk.chunk_id
        
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0.0
            # Create a copy to avoid mutating original
            chunks_by_id[chunk_id] = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                source=chunk.source,
                bm25_score=chunk.bm25_score,
                vector_score=None,
                rrf_score=None,
                rerank_score=None,
                pre_rerank_position=None,
                post_rerank_position=None
            )
        
        rrf_scores[chunk_id] += rrf_score
        # Preserve BM25 score if this chunk came from BM25
        if chunk.bm25_score is not None:
            chunks_by_id[chunk_id].bm25_score = chunk.bm25_score
    
    # Process vector results
    for rank, chunk in enumerate(vector_chunks, start=1):
        rrf_score = 1.0 / (k + rank)
        chunk_id = chunk.chunk_id
        
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0.0
            chunks_by_id[chunk_id] = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                source=chunk.source,
                bm25_score=None,
                vector_score=chunk.vector_score,
                rrf_score=None,
                rerank_score=None,
                pre_rerank_position=None,
                post_rerank_position=None
            )
        
        rrf_scores[chunk_id] += rrf_score
        # Preserve vector score if this chunk came from vector search
        if chunk.vector_score is not None:
            chunks_by_id[chunk_id].vector_score = chunk.vector_score
    
    # Assign RRF scores and sort
    result_chunks = []
    for chunk_id, rrf_score in rrf_scores.items():
        chunk = chunks_by_id[chunk_id]
        chunk.rrf_score = round(rrf_score, 6)
        result_chunks.append(chunk)
    
    # Sort by RRF score descending
    result_chunks.sort(key=lambda c: c.rrf_score or 0.0, reverse=True)
    
    return result_chunks
