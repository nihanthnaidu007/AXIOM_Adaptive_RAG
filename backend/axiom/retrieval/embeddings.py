"""AXIOM Embeddings — cloud (OpenAI) and local (Ollama) transports.

Single seam for every embedding call in the pipeline. The transport is
selected by config (``EMBEDDING_PROVIDER``); ``embed_text`` / ``embed_batch``
signatures are identical for both providers, so callers never branch.

Local mode uses Ollama's native ``/api/embed`` over raw httpx — the same
zero-SDK discipline as the critic and the local generation transport. The
OpenAI-compat shim is deliberately NOT used: it ignores OpenAI's
``dimensions`` kwarg, which would silently produce model-native-width vectors
while the pipeline believes it got ``EMBEDDING_DIMENSIONS``. In local mode the
model's native width is configured explicitly (``LOCAL_EMBEDDING_DIMENSIONS``)
and every response is width-verified (fail-visible, never truncated).
"""

import hashlib
import logging
from typing import List, Optional

import httpx
from cachetools import LRUCache
from openai import AsyncOpenAI

from axiom.config import AxiomConfig, get_config
from axiom.observability.metrics import record_llm_usage
from axiom.provider_mode import LOCAL_PROVIDER

logger = logging.getLogger(__name__)


class EmbeddingDimensionError(RuntimeError):
    """The embedding provider returned a vector of unexpected width.

    Fail-visible by design: silently storing a wrong-width vector would
    poison pgvector inserts (column-dimension enforcement) and cross-contaminate
    the semantic cache with incomparable vectors. Never truncated to fit.
    """


class EmbeddingsClient:
    """Embeddings client behind one seam — OpenAI cloud or Ollama local."""

    def __init__(
        self,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        config: Optional[AxiomConfig] = None,
    ) -> None:
        cfg = config or get_config()
        self._local = cfg.embedding_provider == LOCAL_PROVIDER
        self.model = cfg.effective_embedding_model
        self.dimensions = cfg.effective_embedding_dimensions
        self._cache: LRUCache = LRUCache(maxsize=2000)
        # Exactly one of these is live; the accessor methods narrow before use.
        self._http: Optional[httpx.AsyncClient] = None
        self._openai: Optional[AsyncOpenAI] = None
        if self._local:
            self._base_url = cfg.ollama_host.rstrip("/")
            self._http = httpx.AsyncClient(
                trust_env=True,
                timeout=httpx.Timeout(60.0, connect=5.0),
                transport=transport,
            )
        else:
            self._openai = AsyncOpenAI(api_key=cfg.openai_api_key)

    # ------------------------------------------------------------------
    # Public seam — identical signatures for both providers
    # ------------------------------------------------------------------

    async def embed_text(self, text: str) -> List[float]:
        cache_key = self._cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if self._local:
            (embedding,) = await self._embed_local([text])
        else:
            client = self._openai
            if client is None:  # pragma: no cover — constructor guarantees
                raise RuntimeError("Cloud embeddings client unavailable")
            response = await client.embeddings.create(
                model=self.model, input=text, dimensions=self.dimensions
            )
            self._report_usage(response)
            embedding = response.data[0].embedding
        self._cache[cache_key] = embedding
        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._local:
            embeddings = await self._embed_local(texts)
        else:
            client = self._openai
            if client is None:  # pragma: no cover — constructor guarantees
                raise RuntimeError("Cloud embeddings client unavailable")
            response = await client.embeddings.create(
                model=self.model, input=texts, dimensions=self.dimensions
            )
            self._report_usage(response)
            sorted_data = sorted(response.data, key=lambda x: x.index)
            embeddings = [item.embedding for item in sorted_data]
        for text, emb in zip(texts, embeddings):
            self._cache[self._cache_key(text)] = emb
        return embeddings

    # ------------------------------------------------------------------
    # Local (Ollama) transport — raw httpx, native /api/embed
    # ------------------------------------------------------------------

    async def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """Batch-embed via Ollama's native /api/embed.

        No ``dimensions`` kwarg is sent: Ollama's OpenAI-compat layer ignores
        it, so the model's native width is owned in code
        (LOCAL_EMBEDDING_DIMENSIONS) and verified on every response.
        """
        client = self._http
        if client is None:  # pragma: no cover — constructor guarantees
            raise RuntimeError("Local embeddings client unavailable")
        resp = await client.post(
            f"{self._base_url}/api/embed",
            json={"model": self.model, "input": texts},
        )
        if resp.status_code != 200:
            detail = ""
            try:
                detail = str(resp.json().get("error", ""))
            except ValueError:
                pass
            raise RuntimeError(
                f"Ollama embeddings failed with HTTP {resp.status_code}"
                + (f": {detail}" if detail else "")
            )
        embeddings = resp.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama embeddings response did not cover every input")
        for emb in embeddings:
            if len(emb) != self.dimensions:
                raise EmbeddingDimensionError(
                    f"Local embedding model {self.model!r} returned {len(emb)} dimensions "
                    f"but the pipeline is configured for {self.dimensions}. Set "
                    f"LOCAL_EMBEDDING_DIMENSIONS to the model's native width, then run "
                    f"`alembic upgrade head` and `python -m axiom.reembed`."
                )
        return embeddings

    # ------------------------------------------------------------------
    # Shared plumbing
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        """In-process LRU key — includes model + dimensions so a mid-process
        provider switch can never serve stale-width vectors from the old model."""
        digest = hashlib.md5(text.encode()).hexdigest()
        return f"{self.model}:{self.dimensions}:{digest}"

    def _report_usage(self, response: object) -> None:
        """Report embedding prompt-token usage (embeddings have no completion)."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        try:
            record_llm_usage(int(usage.prompt_tokens), 0)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning("Could not record embedding token usage: %s", exc)


embeddings_client = EmbeddingsClient()


async def embed_text(text: str) -> List[float]:
    return await embeddings_client.embed_text(text)


async def embed_batch(texts: List[str]) -> List[List[float]]:
    return await embeddings_client.embed_batch(texts)
