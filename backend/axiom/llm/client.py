"""LLM client for AXIOM — cloud (Anthropic) and local (Ollama) transports.

All graph nodes import from here. The transport is selected by config
(``LLM_PROVIDER``): the module-level ``chat`` / ``chat_stream`` / ``chat_json``
signatures are identical for both providers, so no call site branches on the
provider. The local transport is raw httpx against Ollama's native ``/api/chat``
(native Ollama, not the OpenAI-compat shim) — the same zero-SDK discipline as
``axiom/evaluation/critic_llm.py`` — keeping the local path free of new
dependencies and of OpenAI-shaped request drift.

Failure semantics (Wave 3, fail-visible):
- Transient conditions (connection errors, 429/5xx, "overloaded") are retried
  with exponential backoff BEFORE the first token reaches a consumer; after
  streaming has begun, retrying would duplicate delivered text, so mid-stream
  failures propagate immediately.
- Non-transient transport failures raise; the caller (``generate_answer``)
  converts them into a visible pipeline failure. A downed generator can never
  yield an answer-shaped string.
"""

import asyncio
import json
import logging
import random
from typing import Any, AsyncIterator, Dict, Optional

import httpx
from anthropic import AsyncAnthropic

from axiom.config import AxiomConfig, get_config
from axiom.observability.metrics import record_llm_usage
from axiom.provider_mode import LOCAL_PROVIDER

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BASE_DELAY_S = 0.75  # tests patch this


class GenerationTransportError(RuntimeError):
    """The configured generation backend failed at the transport level.

    Raised with a message safe for server logs (never embedded in client
    payloads — the generation node replaces it with a sanitized constant).
    """


def _is_transient(status_code: Optional[int], message: str) -> bool:
    """Cloud-side transient classification, mirroring the original logic.

    529 (Anthropic "overloaded") and the 429/5xx families retry; everything
    else fails immediately.
    """
    if status_code is not None and (status_code in _TRANSIENT_STATUS or status_code == 529):
        return True
    lowered = message.lower()
    return "overloaded" in lowered or "connection error" in lowered or "timed out" in lowered


class LLMClient:
    """Generation client behind one seam — Anthropic cloud or Ollama local."""

    def __init__(
        self,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        config: Optional[AxiomConfig] = None,
    ) -> None:
        cfg = config or get_config()
        self._local = cfg.llm_provider == LOCAL_PROVIDER
        self._base_url = cfg.ollama_host.rstrip("/")
        self._default_model = cfg.ollama_generation_model if self._local else cfg.claude_model
        if self._local:
            # Native Ollama /api/chat over raw httpx. Generation on CPU can be
            # slow — a generous total timeout with a tight connect timeout
            # keeps long completions alive while failing fast on dead hosts.
            self._client: Any = httpx.AsyncClient(
                trust_env=True,
                timeout=httpx.Timeout(120.0, connect=5.0),
                transport=transport,
            )
        else:
            # Ensure proxy/network settings from the environment are honored
            # (timeouts on the client default to 600s).
            http_client = httpx.AsyncClient(trust_env=True)
            self._client = AsyncAnthropic(api_key=cfg.anthropic_api_key, http_client=http_client)

    # ------------------------------------------------------------------
    # Public seam — identical signatures for both providers
    # ------------------------------------------------------------------

    async def chat(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> str:
        """Generate a text completion for the prompt (non-streaming)."""
        return await self._chat(prompt, model=model, max_tokens=max_tokens, json_mode=False)

    def chat_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> AsyncIterator[str]:
        """Stream text deltas as they are generated."""
        model = model or self._default_model
        if self._local:
            return self._chat_stream_local(prompt, model, max_tokens)
        return self._chat_stream_cloud(prompt, model, max_tokens)

    async def chat_json(
        self, prompt: str, model: Optional[str] = None, max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Generate a JSON object.

        Cloud: asks Claude for JSON and strips optional fences. Local: Ollama
        runs with ``format: "json"`` (JSON-constrained decoding), so malformed
        output means the model genuinely violated JSON and the ValueError
        surfaces — never a silent string.
        """
        raw = await self._chat(prompt, model=model, max_tokens=max_tokens, json_mode=True)
        text = raw.strip()
        # Strip markdown fences if present (cloud path may wrap JSON).
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        return json.loads(text)  # may raise ValueError — deliberate, visible

    async def aclose(self) -> None:
        """Release the underlying HTTP client (provider-aware)."""
        client = self._client
        if client is None:
            return
        if self._local:
            await client.aclose()
        else:
            client.close()

    async def probe(self) -> bool:
        """One-shot generator availability probe for lifespan startup.

        Local: a tiny generation round-trip confirming the configured model is
        pulled and responsive. Cloud: key presence only — a network probe
        would burn tokens and duplicate what the evaluator already checks.
        """
        if not self._local:
            return bool(get_config().anthropic_api_key)
        try:
            resp = await self._client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._default_model,
                    "prompt": "ping",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=10.0,
            )
            return resp.status_code == 200
        except Exception as exc:  # noqa: BLE001 — probe reports, never raises
            logger.warning("Generator probe failed (%s): %s", type(exc).__name__, exc)
            return False

    # ------------------------------------------------------------------
    # Shared plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        """Exponential backoff with jitter: 0.75s, 1.5s, 3s ±20%."""
        return _RETRY_BASE_DELAY_S * (2**attempt) * random.uniform(0.8, 1.2)

    @staticmethod
    def _report_local_usage(data: Dict[str, Any]) -> None:
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
        if prompt_tokens or completion_tokens:
            record_llm_usage(prompt_tokens, completion_tokens)

    async def _chat(
        self,
        prompt: str,
        *,
        model: Optional[str],
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        model = model or self._default_model
        if self._local:
            return await self._chat_local(prompt, model, max_tokens, json_mode)
        return await self._chat_cloud(prompt, model, max_tokens)

    # ------------------------------------------------------------------
    # Local (Ollama) transport — raw httpx, native /api/chat
    # ------------------------------------------------------------------

    def _chat_payload(
        self, prompt: str, model: str, max_tokens: int, *, stream: bool, json_mode: bool
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "options": {"num_predict": max_tokens},  # max_tokens semantics map to num_predict
        }
        if json_mode:
            payload["format"] = "json"  # JSON-constrained decoding
        return payload

    async def _chat_local(self, prompt: str, model: str, max_tokens: int, json_mode: bool) -> str:
        payload = self._chat_payload(prompt, model, max_tokens, stream=False, json_mode=json_mode)
        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await self._client.post(f"{self._base_url}/api/chat", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = (data.get("message") or {}).get("content")
                    if content is None:
                        raise GenerationTransportError(
                            "Ollama chat response missing message content"
                        )
                    self._report_local_usage(data)
                    return str(content)
                status = resp.status_code
                try:
                    detail = str(resp.json().get("error", ""))
                except ValueError:
                    detail = ""
                if status in _TRANSIENT_STATUS and attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        "Ollama chat transient failure (HTTP %s, attempt %s/%s)%s",
                        status,
                        attempt + 1,
                        _MAX_RETRIES,
                        f": {detail}" if detail else "",
                    )
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                raise GenerationTransportError(
                    f"Ollama chat failed with HTTP {status}" + (f": {detail}" if detail else "")
                )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        "Ollama chat transport error (attempt %s/%s): %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        exc,
                    )
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                logger.error("Ollama chat failed after %s attempts: %s", _MAX_RETRIES, exc)
                raise GenerationTransportError(
                    f"Ollama chat transport error: {type(exc).__name__}"
                ) from exc
        raise GenerationTransportError(
            f"Ollama chat failed after {_MAX_RETRIES} attempts: {last_error}"
        )

    async def _chat_stream_local(
        self, prompt: str, model: str, max_tokens: int
    ) -> AsyncIterator[str]:
        payload = self._chat_payload(prompt, model, max_tokens, stream=True, json_mode=False)
        for attempt in range(_MAX_RETRIES):
            emitted = False
            try:
                async with self._client.stream(
                    "POST", f"{self._base_url}/api/chat", json=payload
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get("error"):
                            raise GenerationTransportError(f"Ollama stream error: {data['error']}")
                        content = (data.get("message") or {}).get("content", "")
                        if content:
                            emitted = True
                            yield content
                        if data.get("done"):
                            self._report_local_usage(data)
                            return
                    return  # stream closed without done frame — treat as complete
            except httpx.HTTPError as exc:
                if emitted:
                    # Mid-stream: retrying would duplicate delivered text.
                    raise
                status = getattr(getattr(exc, "response", None), "status_code", None)
                transient = status in _TRANSIENT_STATUS if status is not None else True
                if transient and attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        "Ollama stream transient failure before first token (attempt %s/%s): %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        exc,
                    )
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                logger.error("Ollama stream failed after %s attempts: %s", _MAX_RETRIES, exc)
                raise GenerationTransportError(
                    f"Ollama stream transport error: {type(exc).__name__}"
                ) from exc

    # ------------------------------------------------------------------
    # Cloud (Anthropic) transport — original behavior preserved
    # ------------------------------------------------------------------

    async def _chat_cloud(self, prompt: str, model: str, max_tokens: int) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                # Extract the text content from the response
                if response.content and len(response.content) > 0:
                    text_block = response.content[0]
                    if hasattr(text_block, "text"):
                        self._report_cloud_usage(response)
                        return text_block.text
                raise GenerationTransportError("Anthropic response contained no text content")
            except Exception as exc:
                if (
                    _is_transient(getattr(exc, "status_code", None), str(exc))
                    and attempt < _MAX_RETRIES - 1
                ):
                    last_error = exc
                    logger.warning(
                        "Anthropic chat transient failure (attempt %s/%s): %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        exc,
                    )
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                raise
        raise GenerationTransportError(
            f"Anthropic chat failed after {_MAX_RETRIES} attempts: {last_error}"
        )

    async def _chat_stream_cloud(
        self, prompt: str, model: str, max_tokens: int
    ) -> AsyncIterator[str]:
        # Retry-on-connect discipline: the stream context may fail before any
        # event is delivered; once events flow, errors propagate (no
        # duplication of already-emitted deltas).
        for attempt in range(_MAX_RETRIES):
            emitted = False
            try:
                async with self._client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    async for text in stream.text_stream:
                        emitted = True
                        yield text
                    return
            except Exception as exc:
                if emitted:
                    raise
                if (
                    _is_transient(getattr(exc, "status_code", None), str(exc))
                    and attempt < _MAX_RETRIES - 1
                ):
                    logger.warning(
                        "Anthropic stream transient failure before first token (attempt %s/%s): %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        exc,
                    )
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                raise

    def _report_cloud_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        completion_tokens = getattr(usage, "output_tokens", 0) or 0
        if prompt_tokens or completion_tokens:
            record_llm_usage(prompt_tokens, completion_tokens)


# Singleton instance
llm_client = LLMClient()


async def chat(prompt: str, model: Optional[str] = None, max_tokens: int = 2000) -> str:
    return await llm_client.chat(prompt, model=model, max_tokens=max_tokens)


def chat_stream(
    prompt: str, model: Optional[str] = None, max_tokens: int = 2000
) -> AsyncIterator[str]:
    """Module-level streaming entry point mirroring ``chat``."""
    return llm_client.chat_stream(prompt, model=model, max_tokens=max_tokens)


async def chat_json(
    prompt: str, model: Optional[str] = None, max_tokens: int = 2000
) -> Dict[str, Any]:
    return await llm_client.chat_json(prompt, model=model, max_tokens=max_tokens)
