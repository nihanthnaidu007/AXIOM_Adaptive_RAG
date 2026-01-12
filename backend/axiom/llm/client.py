"""Clean Anthropic LLM client for AXIOM.

All graph nodes import from here.
"""

import asyncio
import json
import logging
import random
import re

import anthropic
from anthropic import AsyncAnthropic
import httpx

from axiom.config import get_config

logger = logging.getLogger(__name__)

_MARKDOWN_FENCE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)


class LLMClient:
    def __init__(self):
        config = get_config()
        # Ensure proxy/network settings from the environment are honored.
        # This is important in corporate/VPN setups where outbound traffic
        # to `api.anthropic.com` must go through a proxy.
        http_client = httpx.AsyncClient(trust_env=True)
        self._client = AsyncAnthropic(api_key=config.anthropic_api_key, http_client=http_client)
        self._default_model = config.claude_model

    async def chat(self, prompt: str, model: str = None, max_tokens: int = 2000) -> str:
        model = model or self._default_model
        last_exc: Exception | None = None

        # Retry transient service overloads (e.g. 529).
        # This prevents the UI from getting stuck on temporary Anthropic issues.
        # Keep retries short so the graph doesn't feel "stuck"
        # during transient Anthropic overload windows.
        max_retries = 3
        base_delay_s = 0.75
        for attempt in range(max_retries):
            try:
                response = await self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except anthropic.APIError as exc:
                last_exc = exc

                status_code = getattr(exc, "status_code", None)
                msg = str(exc).lower()
                is_overloaded = status_code == 529 or "overloaded" in msg
                is_transient = (
                    is_overloaded
                    or status_code in {429, 500, 502, 503, 504}
                    or "connection error" in msg
                    or "timed out" in msg
                )

                if is_transient and attempt < max_retries - 1:
                    delay = base_delay_s * (2**attempt)
                    # Add a small jitter to avoid synchronized retry bursts.
                    delay *= random.uniform(0.8, 1.2)
                    await asyncio.sleep(delay)
                    continue

                logger.error("Anthropic API error (attempt %s): %s", attempt + 1, exc)
                raise

        # Should never get here, but keeps mypy happy.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Anthropic call failed without an exception.")

    async def chat_json(self, prompt: str, model: str = None, max_tokens: int = 2000) -> dict:
        raw = await self.chat(prompt, model=model, max_tokens=max_tokens)
        text = raw.strip()
        fence_match = _MARKDOWN_FENCE.match(text)
        if fence_match:
            text = fence_match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"LLM response was not valid JSON: {raw}")


llm_client = LLMClient()


async def chat(prompt: str, model: str = None, max_tokens: int = 2000) -> str:
    return await llm_client.chat(prompt, model=model, max_tokens=max_tokens)


async def chat_json(prompt: str, model: str = None, max_tokens: int = 2000) -> dict:
    return await llm_client.chat_json(prompt, model=model, max_tokens=max_tokens)
