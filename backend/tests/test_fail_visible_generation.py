"""Fail-visible generation tests (W3, spec D1).

The pre-W3 node swallowed generation exceptions and stored an answer-shaped
string ("Answer generation failed. Please try again.") — that string then
flowed through evaluation, the cache write, and terminal frames as if it were
a real answer. W3 contract: a failed generation raises
``GenerationFailedError`` (sanitized message; raw detail only in server logs)
so the API surfaces its existing SSE ``error`` frame / HTTP 500 envelope.
"""

from unittest.mock import patch

import pytest

from axiom.graph.nodes.generate_answer import (
    GENERATION_FAILED_MESSAGE,
    GenerationFailedError,
    generate_answer_node,
)
from axiom.graph.state import RetrievedChunk


def _state_with_context() -> dict:
    return {
        "user_query": "What does the spec say about retries?",
        "reranked_chunks": [
            RetrievedChunk(
                chunk_id="c1",
                source="spec.md",
                content="Retries are transient-only and fail visibly.",
                rerank_score=0.9,
            )
        ],
        "web_search_chunks": [],
        "web_search_used": False,
        "correction_attempts": 0,
        "answer_history": [],
        "trace_steps": [],
    }


class TestFailVisibleGeneration:
    @pytest.mark.asyncio
    async def test_generation_failure_raises(self):
        """A downed generator fails the node — no fallback answer string."""
        with patch(
            "axiom.graph.nodes.generate_answer.generate_with_optional_streaming",
            side_effect=RuntimeError("Ollama chat failed with HTTP 503: loading"),
        ):
            with pytest.raises(GenerationFailedError, match="generation failed"):
                await generate_answer_node(dict(_state_with_context()))

    @pytest.mark.asyncio
    async def test_failure_message_is_sanitized_constant(self):
        """The raised message is the sanitized constant — raw exception text
        (model names, hosts, API plans) never becomes the client message."""
        with patch(
            "axiom.graph.nodes.generate_answer.generate_with_optional_streaming",
            side_effect=RuntimeError("secret-hostname.local:11434 plan exceeded"),
        ):
            with pytest.raises(GenerationFailedError) as exc_info:
                await generate_answer_node(dict(_state_with_context()))
            assert "secret-hostname" not in str(exc_info.value)
            assert str(exc_info.value) == GENERATION_FAILED_MESSAGE
