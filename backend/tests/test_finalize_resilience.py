"""finalize_answer_node must never fail a completed query on the cache path.

The cache write (and the query embedding that feeds it) is best-effort: an
embeddings outage must not turn a fully generated, evaluated answer into a
500. Regression for the embed_text-outside-try bug.
"""

from unittest.mock import AsyncMock, patch

import pytest

from axiom.graph.nodes.finalize_answer import finalize_answer_node
from axiom.graph.state import ConfidenceBand


def _evaluated_state():
    return {
        "user_query": "What is the default k value in Reciprocal Rank Fusion?",
        "session_id": "sess-finalize-1",
        "generated_answer": "The default k value in RRF is 60.",
        "served_from_cache": False,
        "cache_result": None,
        "ragas_scores": None,
        "correction_attempts": 0,
        "evaluation_passed": True,
        "trace_steps": [],
    }


@pytest.mark.asyncio
async def test_completes_when_cache_path_embedding_fails():
    state = _evaluated_state()

    async def boom(*args, **kwargs):
        raise RuntimeError("embeddings provider down")

    with patch(
        "axiom.retrieval.embeddings.embed_text", new=AsyncMock(side_effect=boom)
    ):
        result = await finalize_answer_node(state)

    assert result["is_complete"] is True
    assert result["final_answer"] == "The default k value in RRF is 60."
    assert isinstance(result["confidence"], ConfidenceBand)


@pytest.mark.asyncio
async def test_completes_when_cache_store_fails():
    state = _evaluated_state()
    state["query_embedding"] = [0.1, 0.2, 0.3]

    async def boom(*args, **kwargs):
        raise RuntimeError("redis down")

    with patch(
        "axiom.cache.semantic_cache.semantic_cache.store",
        new=AsyncMock(side_effect=boom),
    ):
        result = await finalize_answer_node(state)

    assert result["is_complete"] is True
    assert result["final_answer"] == "The default k value in RRF is 60."


@pytest.mark.asyncio
async def test_writes_cache_when_embedding_succeeds():
    state = _evaluated_state()

    async def ok_embedding(text):
        return [0.4, 0.5, 0.6]

    stored = {}

    async def ok_store(**kwargs):
        stored.update(kwargs)

    with patch(
        "axiom.retrieval.embeddings.embed_text", new=AsyncMock(side_effect=ok_embedding)
    ), patch(
        "axiom.cache.semantic_cache.semantic_cache.store", new=AsyncMock(side_effect=ok_store)
    ):
        result = await finalize_answer_node(state)

    assert result["is_complete"] is True
    assert stored["query_embedding"] == [0.4, 0.5, 0.6]
    assert stored["user_query"] == state["user_query"]
