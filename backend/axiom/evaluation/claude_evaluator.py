"""AXIOM Claude API Evaluator - Cloud-portable RAGAS critic using Anthropic Haiku.

Drop-in replacement for the Ollama-based critic_llm.generate() interface.
Works in every deployment environment (Railway, Render, Fly.io, etc.) because
it uses the Anthropic HTTPS API instead of a local Ollama process.

Hard failures (missing key, unrecoverable error) return empty string. The
RAGAS scorer treats empty as parse failure, and evaluate_answer treats parse
failure as below threshold. Unknown quality is treated as failed quality,
not pretended quality. This is the correct degraded behavior.

Module path: backend/axiom/evaluation/claude_evaluator.py
"""

import asyncio
import logging
from typing import Optional

from anthropic import AsyncAnthropic
from anthropic import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from axiom.config import get_config

logger = logging.getLogger(__name__)

# Module-level model constant. Node files never reference the model string
# directly. They go through claude_evaluator.generate() which uses this.
EVALUATOR_MODEL = "claude-haiku-4-5-20251001"

# Transient HTTP status codes the SDK surfaces via APIStatusError.
# 429 = rate limited. 529 = Anthropic overloaded.
_RETRYABLE_STATUS_CODES = {429, 529}

# One retry on transient failure before giving up. Total worst-case added
# latency per failed eval call: 1.5s. Eval calls run in parallel via
# asyncio.gather in RAGASScorer.score_all, so this is per-prompt, not total.
_RETRY_DELAY_SECONDS = 1.5


class ClaudeEvaluator:
    """Async Anthropic-backed evaluator with one-shot retry on transient errors.

    Exposes the same .generate(prompt, max_tokens) interface as critic_llm so
    that ragas_scorer.py only needs an import swap and constructor change.
    """

    def __init__(self) -> None:
        self._client: Optional[AsyncAnthropic] = None
        self._model: str = EVALUATOR_MODEL

    def _get_client(self) -> AsyncAnthropic:
        """Lazily initialize the client. Raises if ANTHROPIC_API_KEY is empty
        so the error message is clearer than the SDK's generic auth failure.
        """
        if self._client is None:
            cfg = get_config()
            if not cfg.anthropic_api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not configured. "
                    "ClaudeEvaluator cannot run without it."
                )
            self._client = AsyncAnthropic(api_key=cfg.anthropic_api_key)
        return self._client

    async def _call_once(self, prompt: str, max_tokens: int) -> str:
        """Single Anthropic call, no retry. Raises on any SDK exception."""
        client = self._get_client()
        msg = await client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if not msg.content:
            return ""
        first_block = msg.content[0]
        return getattr(first_block, "text", "") or ""

    async def generate(self, prompt: str, max_tokens: int = 300) -> str:
        """Generate evaluator response. Returns empty string on hard failure.

        Retries once on transient failures (429, 529, timeout, connection)
        before returning empty. Non-transient errors fail fast.

        Empty string triggers parse_error in RAGASScorer._parse_score, which
        sets parse_error=True in score_all, which evaluate_answer treats as
        below threshold. UNRELIABLE confidence band. No pretending.
        """
        for attempt in (0, 1):
            try:
                return await self._call_once(prompt, max_tokens)

            except RateLimitError as exc:
                # 429: subclass of APIStatusError, caught first for clarity.
                if attempt == 0:
                    logger.warning(
                        "Claude evaluator rate-limited (429), retrying in %.1fs",
                        _RETRY_DELAY_SECONDS,
                    )
                    await asyncio.sleep(_RETRY_DELAY_SECONDS)
                    continue
                logger.warning("Claude evaluator rate-limited (429), giving up: %s", exc)
                return ""

            except APIStatusError as exc:
                status = getattr(exc, "status_code", None)
                if status in _RETRYABLE_STATUS_CODES and attempt == 0:
                    logger.warning(
                        "Claude evaluator API status %s, retrying in %.1fs",
                        status, _RETRY_DELAY_SECONDS,
                    )
                    await asyncio.sleep(_RETRY_DELAY_SECONDS)
                    continue
                logger.warning(
                    "Claude evaluator API status %s, not retrying: %s", status, exc
                )
                return ""

            except (APITimeoutError, APIConnectionError) as exc:
                if attempt == 0:
                    logger.warning(
                        "Claude evaluator transient network error, retrying in %.1fs: %s",
                        _RETRY_DELAY_SECONDS, exc,
                    )
                    await asyncio.sleep(_RETRY_DELAY_SECONDS)
                    continue
                logger.warning("Claude evaluator network error, giving up: %s", exc)
                return ""

            except Exception as exc:
                # Non-retryable: auth failure, bad request, SDK bug, etc.
                logger.warning(
                    "Claude evaluator generate failed (%s): %s",
                    type(exc).__name__, exc,
                )
                return ""

        return ""  # unreachable, defensive

    async def is_available(self) -> bool:
        """One-shot health probe for lifespan startup logging.

        Returns True only if the API key is configured AND a minimal
        round trip to Anthropic succeeds. Does not retry. A transient
        failure at boot is fine; actual eval calls retry independently.
        """
        try:
            result = await self._call_once("Say OK.", max_tokens=5)
            return bool(result)
        except Exception as exc:
            logger.warning(
                "Claude evaluator availability check failed (%s): %s",
                type(exc).__name__, exc,
            )
            return False


# Module-level singleton, mirrors the critic_llm pattern.
claude_evaluator = ClaudeEvaluator()
