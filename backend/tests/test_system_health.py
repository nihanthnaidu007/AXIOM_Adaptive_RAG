"""Unit tests for Feature 3: system health.

Tests cover:
- _compute_stub_mode derives correctly from _system_health
- /api/health returns system_health with all 5 keys
- QueryResponse includes system_health
- No Ollama call in _compute_stub_mode

No real network calls are made.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


# ---------------------------------------------------------------------------
# _compute_stub_mode tests
# ---------------------------------------------------------------------------

class TestComputeStubMode:
    """_compute_stub_mode must read _system_health without network calls."""

    def _set_health(self, **overrides):
        """Patch _system_health with a full healthy baseline plus overrides."""
        base = {
            "pgvector": "connected",
            "redis": "connected",
            "reranker": "loaded",
            "web_search": "tavily",
            "evaluator": "claude-haiku",
        }
        base.update(overrides)
        return base

    def test_returns_false_when_all_healthy(self):
        health = self._set_health()
        with patch("server._system_health", health):
            from server import _compute_stub_mode
            assert _compute_stub_mode() is False

    def test_returns_true_when_pgvector_not_connected(self):
        health = self._set_health(pgvector="not_connected")
        with patch("server._system_health", health):
            from server import _compute_stub_mode
            assert _compute_stub_mode() is True

    def test_returns_true_when_evaluator_unreachable(self):
        health = self._set_health(evaluator="claude-haiku/unreachable")
        with patch("server._system_health", health):
            from server import _compute_stub_mode
            assert _compute_stub_mode() is True

    def test_returns_true_when_evaluator_unavailable(self):
        health = self._set_health(evaluator="ollama/unavailable")
        with patch("server._system_health", health):
            from server import _compute_stub_mode
            assert _compute_stub_mode() is True

    def test_returns_true_when_reranker_not_loaded(self):
        health = self._set_health(reranker="not_loaded")
        with patch("server._system_health", health):
            from server import _compute_stub_mode
            assert _compute_stub_mode() is True

    def test_returns_false_when_redis_not_connected(self):
        """Redis down degrades cache but does not trigger stub mode."""
        health = self._set_health(redis="not_connected")
        with patch("server._system_health", health):
            from server import _compute_stub_mode
            assert _compute_stub_mode() is False

    def test_returns_false_when_web_search_not_configured(self):
        """No Tavily key is a normal document-only deployment, not stub mode."""
        health = self._set_health(web_search="not_configured")
        with patch("server._system_health", health):
            from server import _compute_stub_mode
            assert _compute_stub_mode() is False

    def test_returns_true_when_evaluator_unknown(self):
        """Unknown evaluator state means startup failed to assess it."""
        health = self._set_health(evaluator="unknown")
        with patch("server._system_health", health):
            from server import _compute_stub_mode
            assert _compute_stub_mode() is True

    def test_is_not_a_coroutine(self):
        """_compute_stub_mode must be synchronous after Feature 3 fix."""
        import inspect
        from server import _compute_stub_mode
        assert not inspect.iscoroutinefunction(_compute_stub_mode), (
            "_compute_stub_mode must not be async — it reads _system_health directly"
        )

    def test_does_not_call_critic_llm(self):
        """Ollama must not be contacted during stub_mode computation."""
        health = self._set_health()
        with patch("server._system_health", health):
            with patch("server.critic_llm") as mock_ollama:
                from server import _compute_stub_mode
                _compute_stub_mode()
                mock_ollama.is_connected.assert_not_called()


# ---------------------------------------------------------------------------
# system_health field in QueryResponse
# ---------------------------------------------------------------------------

class TestSystemHealthInQueryResponse:
    """QueryResponse must include system_health with all 5 keys."""

    def test_query_response_has_system_health_field(self):
        from server import QueryResponse
        fields = QueryResponse.model_fields
        assert "system_health" in fields, (
            "system_health field missing from QueryResponse"
        )

    def test_system_health_default_is_empty_dict(self):
        from server import QueryResponse
        field = QueryResponse.model_fields["system_health"]
        # Default factory should produce an empty dict
        instance = QueryResponse(
            session_id="test",
            final_answer="test",
            confidence=None,
            classification=None,
            retrieval_strategy="",
            ragas_scores=None,
            scores_history=[],
            reranked_chunks=[],
            correction_attempts=0,
            correction_history=[],
            trace_steps=[],
            served_from_cache=False,
            is_complete=True,
            error=None,
            web_search_used=False,
            web_search_chunks=[],
            document_chunk_count=0,
            web_chunk_count=0,
        )
        assert isinstance(instance.system_health, dict)


# ---------------------------------------------------------------------------
# /api/health endpoint contract
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """GET /api/health must return system_health with all 5 keys and no ollama key."""

    @pytest.mark.asyncio
    async def test_health_returns_system_health(self):
        from httpx import AsyncClient, ASGITransport
        from server import app

        full_health = {
            "pgvector": "connected",
            "redis": "connected",
            "reranker": "loaded",
            "web_search": "not_configured",
            "evaluator": "claude-haiku",
        }

        with patch("server._system_health", full_health):
            with patch("server._compute_stub_mode", return_value=False):
                with patch("server.vector_store") as mock_vs:
                    mock_vs.is_connected = AsyncMock(return_value=True)
                    mock_vs.count = AsyncMock(return_value=0)
                    with patch("server.semantic_cache") as mock_cache:
                        mock_cache.is_connected = AsyncMock(return_value=True)
                        async with AsyncClient(
                            transport=ASGITransport(app=app), base_url="http://test"
                        ) as client:
                            response = await client.get("/api/health")

        assert response.status_code == 200
        data = response.json()

        # system_health must be present with all 5 keys
        sh = data.get("system_health", {})
        expected_keys = {"pgvector", "redis", "reranker", "web_search", "evaluator"}
        missing = expected_keys - set(sh.keys())
        assert not missing, f"Missing keys in system_health: {missing}"

        # services block must not contain ollama key
        svc = data.get("services", {})
        assert "ollama" not in svc, (
            "services block must not contain 'ollama' key after Feature 1 migration"
        )
