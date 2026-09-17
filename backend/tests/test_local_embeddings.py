"""Wave 3 local-embeddings tests (spec D2.4).

httpx-level mocked transports only — no Ollama daemon is contacted. Verified
here: native /api/embed batch shape with NO dimensions kwarg, dimension-honest
verification (wrong width raises, never truncates), batch ordering, the
model+dimensions LRU key, and the preserved cloud path.
"""

import json

import httpx
import pytest

import axiom.retrieval.embeddings as emb_module
from axiom.config import AxiomConfig
from axiom.retrieval.embeddings import EmbeddingDimensionError, EmbeddingsClient

_LOCAL_CFG = AxiomConfig(
    llm_provider="local",
    embedding_provider="local",
    ollama_host="http://ollama.test",
    ollama_embedding_model="nomic-embed-text",
    anthropic_api_key="",
    openai_api_key="",
    postgres_url="postgresql://t",
)

_CLOUD_CFG = AxiomConfig(
    anthropic_api_key="k",
    openai_api_key="sk-x",
    postgres_url="postgresql://t",
)


def _local_client(handler) -> EmbeddingsClient:
    return EmbeddingsClient(transport=httpx.MockTransport(handler), config=_LOCAL_CFG)


class TestLocalEmbed:
    @pytest.mark.asyncio
    async def test_embed_text_posts_native_batch_without_dimensions(self):
        """The request must be native Ollama /api/embed with input=[text] and
        NO dimensions key — the OpenAI-compat shim ignores it (pre-flight R)."""
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3] + [0.0] * 765]})

        out = await _local_client(handler).embed_text("hello")
        assert len(out) == 768 and out[0] == 0.1
        assert seen["path"] == "/api/embed"
        assert seen["body"]["model"] == "nomic-embed-text"
        assert seen["body"]["input"] == ["hello"]
        assert "dimensions" not in seen["body"]

    @pytest.mark.asyncio
    async def test_embed_batch_preserves_order(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["input"] == ["a", "b", "c"]
            return httpx.Response(
                200,
                json={"embeddings": [[float(i)] + [0.0] * 767 for i in range(3)]},
            )

        out = await _local_client(handler).embed_batch(["a", "b", "c"])
        assert [e[0] for e in out] == [0.0, 1.0, 2.0]

    @pytest.mark.asyncio
    async def test_wrong_width_raises_dimension_error_not_truncated(self):
        """A 3-width response against a 768-dim config fails visibly."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})

        client = _local_client(handler)
        with pytest.raises(EmbeddingDimensionError, match="768"):
            await client.embed_text("hello")

    @pytest.mark.asyncio
    async def test_batch_width_mismatch_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": [[0.1] * 768, [0.2] * 3]})

        client = _local_client(handler)
        with pytest.raises(EmbeddingDimensionError):
            await client.embed_batch(["a", "b"])

    @pytest.mark.asyncio
    async def test_short_response_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": []})

        client = _local_client(handler)
        with pytest.raises(RuntimeError, match="did not cover"):
            await client.embed_text("hello")

    @pytest.mark.asyncio
    async def test_http_error_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "model not found"})

        client = _local_client(handler)
        with pytest.raises(RuntimeError, match="404"):
            await client.embed_text("hello")

    @pytest.mark.asyncio
    async def test_lru_cache_key_includes_model_and_dimensions(self):
        """A mid-process provider switch must not serve stale-width vectors."""
        client = _local_client(lambda req: httpx.Response(200, json={"embeddings": [[0.0] * 768]}))
        key = client._cache_key("hello")
        assert key == "nomic-embed-text:768:" + client._cache_key("hello").rsplit(":", 1)[1]


class TestCloudEmbeddingsPreserved:
    @pytest.mark.asyncio
    async def test_cloud_client_uses_openai_shim_unchanged(self, monkeypatch):
        captured = {}

        class _Data:
            index = 0
            embedding = [0.5]

        class _Usage:
            prompt_tokens = 7

        class _Response:
            data = [_Data()]
            usage = _Usage()

        class _Embeddings:
            async def create(self, **kwargs):
                captured.update(kwargs)
                return _Response()

        class _StubOpenAI:
            def __init__(self, **kwargs):
                pass

            embeddings = _Embeddings()

        monkeypatch.setattr(emb_module, "AsyncOpenAI", _StubOpenAI)
        client = EmbeddingsClient(config=_CLOUD_CFG)
        out = await client.embed_text("hello")
        assert out == [0.5]
        assert captured["model"] == "text-embedding-3-small"
        assert captured["dimensions"] == 1536
