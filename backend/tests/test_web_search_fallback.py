"""Unit tests for Feature 2: web search fallback.

Tests cover:
- web_search.py: empty key guard, result normalization, error return
- web_search_node.py: state writes, trace step, unconfigured key path
- graph routing: zero-chunk short-circuit, FACTUAL skip, web-search-used guard,
  post-correction trigger
- generate_answer.py: document-only vs web-augmented context path

No real Anthropic or Tavily API calls are made.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# web_search.py tests
# ---------------------------------------------------------------------------

class TestTavilySearch:
    """tavily_search() must return [] on missing key and normalize results."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_api_key_missing(self):
        with patch("axiom.search.web_search.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(tavily_api_key="")
            from axiom.search.web_search import tavily_search
            result = await tavily_search("test query")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_tavily_exception(self):
        mock_client = MagicMock()
        mock_client.search = AsyncMock(side_effect=Exception("network error"))

        with patch("axiom.search.web_search.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(
                tavily_api_key="test-key",
                tavily_search_depth="basic",
                tavily_max_results=5,
            )
            with patch("axiom.search.web_search.AsyncTavilyClient", return_value=mock_client):
                from axiom.search.web_search import tavily_search
                result = await tavily_search("test query")
        assert result == []

    @pytest.mark.asyncio
    async def test_normalizes_results_correctly(self):
        raw_results = {
            "results": [
                {"url": "https://example.com", "title": "Example", "content": "Some content", "score": 0.9},
                {"url": "https://other.com", "title": "", "content": "  ", "score": 0.1},  # empty content stripped
            ]
        }
        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value=raw_results)

        with patch("axiom.search.web_search.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(
                tavily_api_key="test-key",
                tavily_search_depth="basic",
                tavily_max_results=5,
            )
            with patch("axiom.search.web_search.AsyncTavilyClient", return_value=mock_client):
                from axiom.search.web_search import tavily_search
                result = await tavily_search("test query")

        # Empty content result should be filtered
        assert len(result) == 1
        assert result[0]["url"] == "https://example.com"
        assert result[0]["content"] == "Some content"
        assert result[0]["score"] == 0.9

    def test_is_tavily_configured_returns_false_when_empty(self):
        with patch("axiom.search.web_search.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(tavily_api_key="")
            from axiom.search.web_search import is_tavily_configured
            assert is_tavily_configured() is False

    def test_is_tavily_configured_returns_true_when_set(self):
        with patch("axiom.search.web_search.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(tavily_api_key="tvly-abc123")
            from axiom.search.web_search import is_tavily_configured
            assert is_tavily_configured() is True


# ---------------------------------------------------------------------------
# web_search_node.py tests
# ---------------------------------------------------------------------------

class TestWebSearchNode:
    """web_search_node must always set web_search_used=True and write trace step."""

    @pytest.mark.asyncio
    async def test_sets_web_search_used_on_success(self):
        mock_results = [{"url": "http://x.com", "title": "X", "content": "Content", "score": 0.8}]

        with patch("axiom.graph.nodes.web_search_node.is_tavily_configured", return_value=True):
            with patch("axiom.graph.nodes.web_search_node.tavily_search", new_callable=AsyncMock, return_value=mock_results):
                with patch("axiom.graph.nodes.web_search_node._corpus_is_empty", new_callable=AsyncMock, return_value=False):
                    from axiom.graph.nodes.web_search_node import web_search_node
                    state = {
                        "user_query": "test",
                        "active_query": "test",
                        "web_search_used": False,
                        "web_search_chunks": [],
                        "trace_steps": [],
                    }
                    result = await web_search_node(state)

        assert result["web_search_used"] is True
        assert len(result["web_search_chunks"]) == 1
        assert len(result["trace_steps"]) == 1
        assert result["trace_steps"][0].node_name == "web_search"

    @pytest.mark.asyncio
    async def test_sets_web_search_used_even_when_tavily_not_configured(self):
        with patch("axiom.graph.nodes.web_search_node.is_tavily_configured", return_value=False):
            from axiom.graph.nodes.web_search_node import web_search_node
            state = {
                "user_query": "test",
                "active_query": "test",
                "web_search_used": False,
                "web_search_chunks": [],
                "trace_steps": [],
            }
            result = await web_search_node(state)

        assert result["web_search_used"] is True
        assert result["web_search_chunks"] == []
        assert result["trace_steps"][0].status == "skipped"

    @pytest.mark.asyncio
    async def test_uses_advanced_depth_when_corpus_is_empty(self):
        captured_depth = {}

        async def mock_search(query, search_depth=None, max_results=None):
            captured_depth["depth"] = search_depth
            return []

        with patch("axiom.graph.nodes.web_search_node.is_tavily_configured", return_value=True):
            with patch("axiom.graph.nodes.web_search_node.tavily_search", side_effect=mock_search):
                with patch("axiom.graph.nodes.web_search_node._corpus_is_empty", new_callable=AsyncMock, return_value=True):
                    from axiom.graph.nodes.web_search_node import web_search_node
                    state = {
                        "user_query": "test",
                        "active_query": "test",
                        "web_search_used": False,
                        "web_search_chunks": [],
                        "trace_steps": [],
                    }
                    await web_search_node(state)

        assert captured_depth["depth"] == "advanced", (
            f"Expected advanced depth for empty corpus, got {captured_depth['depth']}"
        )

    @pytest.mark.asyncio
    async def test_uses_basic_depth_when_corpus_has_documents(self):
        captured_depth = {}

        async def mock_search(query, search_depth=None, max_results=None):
            captured_depth["depth"] = search_depth
            return []

        with patch("axiom.graph.nodes.web_search_node.is_tavily_configured", return_value=True):
            with patch("axiom.graph.nodes.web_search_node.tavily_search", side_effect=mock_search):
                with patch("axiom.graph.nodes.web_search_node._corpus_is_empty", new_callable=AsyncMock, return_value=False):
                    from axiom.graph.nodes.web_search_node import web_search_node
                    state = {
                        "user_query": "test",
                        "active_query": "test",
                        "web_search_used": False,
                        "web_search_chunks": [],
                        "trace_steps": [],
                    }
                    await web_search_node(state)

        assert captured_depth["depth"] == "basic"


# ---------------------------------------------------------------------------
# Graph routing function tests
# ---------------------------------------------------------------------------

class TestRouteEvaluation:
    """_route_evaluation must not infinite-loop and must skip FACTUAL."""

    def _make_state(self, **kwargs):
        base = {
            "evaluation_passed": False,
            "correction_attempts": 0,
            "web_search_used": False,
            "classification": {"query_type": "ABSTRACT"},
        }
        base.update(kwargs)
        return base

    def _get_router(self):
        from axiom.graph.graph import _route_evaluation
        return _route_evaluation

    def test_returns_finalize_when_evaluation_passed(self):
        state = self._make_state(evaluation_passed=True)
        assert self._get_router()(state) == "finalize_answer"

    def test_returns_rewrite_when_attempts_below_max(self):
        state = self._make_state(correction_attempts=1)
        with patch("axiom.graph.graph.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(max_correction_attempts=3)
            result = self._get_router()(state)
        assert result == "rewrite_query"

    def test_returns_web_search_when_attempts_exhausted_and_not_factual(self):
        state = self._make_state(
            correction_attempts=3,
            web_search_used=False,
            classification={"query_type": "ABSTRACT"},
        )
        with patch("axiom.graph.graph.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(max_correction_attempts=3)
            result = self._get_router()(state)
        assert result == "web_search"

    def test_returns_finalize_not_web_search_when_web_already_used(self):
        """Critical: prevents infinite loop after web search."""
        state = self._make_state(
            correction_attempts=3,
            web_search_used=True,
            classification={"query_type": "ABSTRACT"},
        )
        with patch("axiom.graph.graph.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(max_correction_attempts=3)
            result = self._get_router()(state)
        assert result == "finalize_answer"

    def test_factual_query_never_routes_to_web_search(self):
        """FACTUAL queries skip web search even with exhausted corrections."""
        state = self._make_state(
            correction_attempts=3,
            web_search_used=False,
            classification={"query_type": "FACTUAL"},
        )
        with patch("axiom.graph.graph.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(max_correction_attempts=3)
            result = self._get_router()(state)
        assert result == "finalize_answer"


class TestRouteFromRerankWithWeb:
    """_route_from_rerank_with_web must short-circuit to web_search on zero chunks."""

    def _get_router(self):
        from axiom.graph.graph import _route_from_rerank_with_web
        return _route_from_rerank_with_web

    def test_zero_chunks_routes_to_web_search(self):
        state = {
            "reranked_chunks": [],
            "web_search_used": False,
            "decomposed": False,
            "classification": {"query_type": "ABSTRACT"},
        }
        assert self._get_router()(state) == "web_search"

    def test_zero_chunks_factual_routes_to_generate_not_web(self):
        state = {
            "reranked_chunks": [],
            "web_search_used": False,
            "decomposed": False,
            "classification": {"query_type": "FACTUAL"},
        }
        assert self._get_router()(state) == "generate_answer"

    def test_zero_chunks_web_already_used_routes_to_generate(self):
        state = {
            "reranked_chunks": [],
            "web_search_used": True,
            "decomposed": False,
            "classification": {"query_type": "ABSTRACT"},
        }
        assert self._get_router()(state) == "generate_answer"

    def test_nonzero_chunks_routes_to_generate(self):
        state = {
            "reranked_chunks": [MagicMock()],
            "web_search_used": False,
            "decomposed": False,
            "classification": {"query_type": "ABSTRACT"},
        }
        assert self._get_router()(state) == "generate_answer"

    def test_decomposed_routes_to_evaluate(self):
        state = {
            "reranked_chunks": [MagicMock()],
            "web_search_used": False,
            "decomposed": True,
            "classification": {"query_type": "MULTI_HOP"},
        }
        assert self._get_router()(state) == "evaluate_answer"
