"""Wave 3 semantic-cache dimension-versioning tests (spec D2.4).

Cache keys are embedding-identity versioned: entries written under one
model/dimension must be unreachable from another. These tests verify the key
geometry without a live Redis (key/tag functions) and the Tier-2 width guard
with a minimal fake redis client.
"""

import pytest

import axiom.cache.semantic_cache as sc_module
from axiom.cache.semantic_cache import SemanticCache
from axiom.config import AxiomConfig


def _patch_config(monkeypatch: pytest.MonkeyPatch, **kwargs) -> None:
    cfg = AxiomConfig(
        anthropic_api_key="k",
        openai_api_key="sk-x",
        postgres_url="postgresql://t",
        **kwargs,
    )
    monkeypatch.setattr(sc_module, "get_config", lambda: cfg)


class TestCacheKeyVersioning:
    def test_cloud_key_carries_model_and_dimensions(self, monkeypatch):
        _patch_config(monkeypatch)
        key = SemanticCache._cache_key("why is caching hard?")
        assert key.startswith("axiom:cache:text-embedding-3-small@1536:")
        assert len(key.rsplit(":", 1)[1]) == 12

    def test_keys_are_deterministic(self, monkeypatch):
        _patch_config(monkeypatch)
        assert (
            SemanticCache._cache_key("q")
            == SemanticCache._cache_key("q")
            != SemanticCache._cache_key("r")
        )

    def test_different_dimensions_produce_different_keys(self, monkeypatch):
        _patch_config(monkeypatch)
        cloud_key = SemanticCache._cache_key("same query")
        _patch_config(
            monkeypatch,
            llm_provider="local",
            embedding_provider="local",
            ollama_embedding_model="nomic-embed-text",
            local_embedding_dimensions=768,
        )
        local_key = SemanticCache._cache_key("same query")
        assert local_key != cloud_key
        assert "nomic-embed-text@768" in local_key

    def test_index_keys_are_versioned(self, monkeypatch):
        _patch_config(monkeypatch)
        assert SemanticCache._zindex_key() == ("axiom:cache:zindex:text-embedding-3-small@1536")
        assert SemanticCache._index_key() == ("axiom:cache:index:text-embedding-3-small@1536")


class TestTier2WidthGuard:
    @pytest.mark.asyncio
    async def test_mismatched_width_entry_is_skipped(self, monkeypatch):
        """A stale-width stored embedding must never enter cosine comparison."""
        _patch_config(monkeypatch)
        cache = SemanticCache()

        class FakeRedis:
            def __init__(self):
                self.calls = []

            async def exists(self, key):
                return 0

            async def zrevrangebyscore(self, *args, **kwargs):
                self.calls.append("zrevrangebyscore")
                return ["axiom:cache:stale:abc"]

            async def smembers(self, key):
                return set()

            def pipeline(self, transaction=False):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

            def hget(self, key, field):
                # 3-wide vector stored under a 1536-dim config
                self._payload = "[1.0, 0.0, 0.0]"
                return self

            async def execute(self):
                return [self._payload]

            async def hgetall(self, key):
                return {}

            async def hincrby(self, key, field, amount):
                return 1

        fake = FakeRedis()
        cache._redis = fake  # type: ignore[assignment]
        cache._connected = True

        result = await cache.search("same query", [0.1] * 1536)
        assert result is None  # stale-width entry skipped → no hit
        assert fake.calls == ["zrevrangebyscore"]
