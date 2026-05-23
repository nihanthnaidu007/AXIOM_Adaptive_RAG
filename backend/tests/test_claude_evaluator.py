"""Unit tests for ClaudeEvaluator.

All tests mock the Anthropic SDK. No real API calls are made.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def evaluator():
    """Fresh ClaudeEvaluator instance with a cleared client for each test."""
    from axiom.evaluation.claude_evaluator import ClaudeEvaluator
    e = ClaudeEvaluator()
    e._client = None  # ensure lazy init does not reuse a previous client
    return e


class TestGenerateReturnsEmptyOnFailure:
    """generate() must return empty string on any unrecoverable error."""

    @pytest.mark.asyncio
    async def test_returns_empty_on_generic_exception(self, evaluator):
        with patch.object(evaluator, "_call_once", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = Exception("unexpected error")
            result = await evaluator.generate("test prompt", max_tokens=50)
        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_empty_when_content_is_empty_list(self, evaluator):
        mock_msg = MagicMock()
        mock_msg.content = []
        with patch.object(evaluator, "_get_client") as mock_client_getter:
            mock_anthropic = MagicMock()
            mock_anthropic.messages.create = AsyncMock(return_value=mock_msg)
            mock_client_getter.return_value = mock_anthropic
            result = await evaluator.generate("test prompt", max_tokens=50)
        assert result == ""


class TestGenerateRetriesOnTransientErrors:
    """generate() must retry once on 429/529/timeout/connection errors."""

    @pytest.mark.asyncio
    async def test_retries_once_on_rate_limit_then_succeeds(self, evaluator):
        from anthropic import RateLimitError

        # First call raises 429, second call succeeds.
        call_results = [
            RateLimitError("rate limited", response=MagicMock(), body={}),
            "real score response",
        ]
        call_count = {"n": 0}

        async def side_effect(prompt, max_tokens):
            result = call_results[call_count["n"]]
            call_count["n"] += 1
            if isinstance(result, Exception):
                raise result
            return result

        with patch.object(evaluator, "_call_once", side_effect=side_effect):
            with patch("axiom.evaluation.claude_evaluator.asyncio.sleep", new_callable=AsyncMock):
                result = await evaluator.generate("test prompt", max_tokens=100)

        assert result == "real score response"
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_returns_empty_after_two_rate_limit_failures(self, evaluator):
        from anthropic import RateLimitError

        with patch.object(
            evaluator,
            "_call_once",
            new_callable=AsyncMock,
            side_effect=RateLimitError("rate limited", response=MagicMock(), body={}),
        ):
            with patch("axiom.evaluation.claude_evaluator.asyncio.sleep", new_callable=AsyncMock):
                result = await evaluator.generate("test prompt", max_tokens=100)

        assert result == ""


class TestIsAvailable:
    """is_available() must reflect real reachability, not assumptions."""

    @pytest.mark.asyncio
    async def test_returns_true_when_call_succeeds(self, evaluator):
        with patch.object(evaluator, "_call_once", new_callable=AsyncMock, return_value="OK"):
            result = await evaluator.is_available()
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_call_raises(self, evaluator):
        with patch.object(
            evaluator, "_call_once", new_callable=AsyncMock, side_effect=Exception("boom")
        ):
            result = await evaluator.is_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_call_returns_empty(self, evaluator):
        with patch.object(evaluator, "_call_once", new_callable=AsyncMock, return_value=""):
            result = await evaluator.is_available()
        assert result is False


class TestMissingApiKey:
    """_get_client() must raise immediately if ANTHROPIC_API_KEY is empty."""

    def test_raises_runtime_error_on_empty_key(self, evaluator):
        with patch("axiom.evaluation.claude_evaluator.get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(anthropic_api_key="")
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not configured"):
                evaluator._get_client()
