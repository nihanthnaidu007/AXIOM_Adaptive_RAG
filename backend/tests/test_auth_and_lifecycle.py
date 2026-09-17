"""Integration tests for Wave 0: fail-closed auth + data lifecycle.

Covers:
- require_api_key fails closed: 503 with guidance when API_KEY is unset,
  401 on missing/wrong key, on every protected endpoint
- 200 with a valid key (stubbed pipeline — no real LLM/embedding calls)
- Content-hash chunk ids: same content -> stable ids, changed content -> new ids
- Re-ingesting an updated file replaces stale chunks (no ON CONFLICT survivors)
- DELETE /api/documents/{doc_id} removes chunk embeddings + lineage
- Semantic cache is invalidated on ingest and delete

Auth tests need no external services. Lifecycle tests connect directly to
PostgreSQL (pgvector) / Redis and skip when those are unavailable (the CI
workflow provisions both).
"""

import uuid
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

API_KEY = "test-api-key"  # keep in sync with tests/conftest.py
AUTH = {"X-API-Key": API_KEY}

TEST_SESSION = "00000000-0000-0000-0000-000000000000"

# Every route guarded by require_api_key. Ingest is multipart, covered separately.
PROTECTED_ENDPOINTS = [
    ("GET", "/api/trace/" + TEST_SESSION, None),
    ("GET", "/api/stats", None),
    ("GET", f"/api/session/{TEST_SESSION}/state", None),
    ("GET", "/api/eval/status/does-not-exist", None),
    ("GET", "/api/eval/results", None),
    ("POST", "/api/eval/run", None),
    ("POST", "/api/eval/run/stream", None),
    ("POST", "/api/query", {"query": "auth probe"}),
    ("POST", "/api/query/stream", {"query": "auth probe"}),
    ("DELETE", "/api/documents/does-not-exist", None),
]


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """The slowapi limiter is process-global and keyed by client address;
    without a reset the 5/minute ingest budget would leak across tests."""
    import server as server_module

    server_module.limiter.reset()
    yield


@pytest_asyncio.fixture(autouse=True)
async def _fresh_singleton_connections():
    """Dispose store singletons after each test.

    asyncpg pool handles and the redis client are bound to the event loop
    that created them; carried into the next test's loop they fail with
    'attached to a different loop'. Each test therefore starts disconnected
    and connects in its own loop.
    """
    yield
    from axiom.cache.semantic_cache import semantic_cache
    from axiom.retrieval.vector_store import vector_store

    engine = vector_store._engine
    if engine is not None:
        await engine.dispose()
        vector_store._engine = None
        vector_store._connected = False
    if semantic_cache._redis is not None:
        await semantic_cache._redis.aclose()
        semantic_cache._redis = None
        semantic_cache._connected = False


@pytest_asyncio.fixture()
async def client():
    from server import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture()
def fake_embeddings(monkeypatch):
    """Stub OpenAI embeddings with deterministic vectors of the configured size."""
    from axiom.config import get_config

    dims = get_config().embedding_dimensions

    async def _fake_embed(texts):
        return [[0.01] * dims for _ in texts]

    monkeypatch.setattr("axiom.ingest.indexer.embed_batch", _fake_embed)


@pytest_asyncio.fixture()
async def connected_services():
    """Connect the store singletons directly (no app lifespan — that would
    load models and connect the checkpointer). Skips if PostgreSQL is absent."""
    from sqlalchemy import text as sa_text

    from axiom.retrieval.vector_store import vector_store

    if not await vector_store.connect():
        pytest.skip("PostgreSQL/pgvector not available — lifecycle tests require it")
    from axiom.cache.semantic_cache import semantic_cache

    await semantic_cache.connect()
    # vector_store.connect() self-creates chunk_embeddings but not the lineage
    # table — the app gets it from startup/alembic, neither of which runs under
    # the test harness. Mirror the initial migration DDL so fresh CI databases
    # have the full schema.
    async with vector_store._engine.begin() as conn:
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS ingested_documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT,
                chunk_count INTEGER,
                file_size_bytes INTEGER,
                indexed_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
    yield


@pytest_asyncio.fixture()
async def corpus_cleanup(connected_services):
    """Record ingested sources and purge their rows + lineage after the test."""
    from sqlalchemy import text as sa_text

    import server as server_module
    from axiom.retrieval.vector_store import get_engine

    sources: list = []
    yield sources
    engine = get_engine()
    if engine:
        async with engine.begin() as conn:
            for src in sources:
                await conn.execute(
                    sa_text("DELETE FROM chunk_embeddings WHERE source = :s"), {"s": src}
                )
                await conn.execute(
                    sa_text("DELETE FROM ingested_documents WHERE filename = :s"), {"s": src}
                )
    server_module._ingested_docs[:] = [
        d for d in server_module._ingested_docs if d.get("filename") not in set(sources)
    ]


async def _ingest(client, filename: str, content: bytes):
    return await client.post(
        "/api/ingest",
        files={"file": (filename, content, "text/plain")},
        headers=AUTH,
    )


# ---------------------------------------------------------------------------
# Fail-closed auth
# ---------------------------------------------------------------------------


class TestFailClosedAuth:
    @pytest.mark.asyncio
    async def test_503_with_guidance_when_api_key_unset(self, client, monkeypatch):
        import server as server_module

        class _UnsetKeyConfig:
            api_key = ""

        monkeypatch.setattr(server_module, "get_config", lambda: _UnsetKeyConfig())

        # Even a matching-format key cannot bypass: there is nothing to match.
        for headers in (None, {"X-API-Key": "anything"}):
            r = await client.get("/api/stats", headers=headers)
            assert r.status_code == 503
            detail = r.json()["detail"]
            assert "API_KEY" in detail["guidance"]
            assert detail["error"] == "API authentication is not configured"

    @pytest.mark.parametrize(("method", "url", "json_body"), PROTECTED_ENDPOINTS)
    @pytest.mark.asyncio
    async def test_missing_key_is_401(self, client, method, url, json_body):
        r = await client.request(method, url, json=json_body)
        assert r.status_code == 401, f"{method} {url} answered {r.status_code} without a key"
        assert r.json()["detail"] == {"error": "Invalid API key"}

    @pytest.mark.parametrize(("method", "url", "json_body"), PROTECTED_ENDPOINTS)
    @pytest.mark.asyncio
    async def test_wrong_key_is_401(self, client, method, url, json_body):
        r = await client.request(
            method, url, json=json_body, headers={"X-API-Key": "wrong-key"}
        )
        assert r.status_code == 401, f"{method} {url} answered {r.status_code} with a wrong key"

    @pytest.mark.asyncio
    async def test_ingest_without_key_is_401(self, client):
        r = await client.post(
            "/api/ingest",
            files={"file": ("probe.txt", b"hello", "text/plain")},
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_health_stays_public(self, client):
        """/api/health must remain reachable for orchestration probes."""
        r = await client.get("/api/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# 200 with a valid key (stubbed pipeline)
# ---------------------------------------------------------------------------


class TestAuthenticatedAccess:
    @pytest.mark.asyncio
    async def test_stats_with_key_is_200(self, client):
        r = await client.get("/api/stats", headers=AUTH)
        assert r.status_code == 200
        body = r.json()
        for key in ("indexed_documents", "bm25_doc_count", "cache_entries", "stub_mode"):
            assert key in body

    @pytest.mark.asyncio
    async def test_trace_with_key_is_200(self, client, monkeypatch):
        import server as server_module

        seeded = [{"node_name": "classify_query", "status": "ok", "summary": "probe"}]
        monkeypatch.setattr(
            server_module, "_trace_store", {TEST_SESSION: seeded}, raising=False
        )
        r = await client.get(f"/api/trace/{TEST_SESSION}", headers=AUTH)
        assert r.status_code == 200
        assert r.json()["trace_steps"] == seeded

    @pytest.mark.asyncio
    async def test_eval_status_with_key_is_200(self, client, monkeypatch):
        import server as server_module

        monkeypatch.setattr(
            server_module,
            "_eval_jobs",
            {"w0-job": {"status": "complete", "progress": 1, "total": 1}},
            raising=False,
        )
        r = await client.get("/api/eval/status/w0-job", headers=AUTH)
        assert r.status_code == 200
        assert r.json()["status"] == "complete"


# ---------------------------------------------------------------------------
# Content-hash chunk ids
# ---------------------------------------------------------------------------


class TestChunkIdContentHash:
    def _chunk(self, source: str, text: str):
        from axiom.ingest.loader import DocumentChunker

        chunker = DocumentChunker()
        pages = chunker.load_text(text, source)
        return chunker.chunk(pages, source=source)

    def test_same_content_yields_stable_ids(self):
        text = "Stable content for hashing. " * 40
        first = self._chunk("doc.txt", text)
        second = self._chunk("doc.txt", text)
        assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]
        assert first, "chunker produced no chunks — test content too short"

    def test_changed_content_yields_new_ids(self):
        source = "doc.txt"
        before = self._chunk(source, "Original version of the text. " * 40)
        after = self._chunk(source, "Updated version of the text. " * 40)
        before_ids = {c["chunk_id"] for c in before}
        after_ids = {c["chunk_id"] for c in after}
        assert before and after
        assert before_ids.isdisjoint(after_ids), (
            "chunk ids must change when file content changes, "
            "otherwise ON CONFLICT DO NOTHING keeps stale chunks"
        )


# ---------------------------------------------------------------------------
# Data lifecycle (requires PostgreSQL; Redis-backed assertions skip without it)
# ---------------------------------------------------------------------------


class TestReingestReplacesChunks:
    @pytest.mark.asyncio
    async def test_reingest_replaces_stale_chunks(
        self, client, connected_services, corpus_cleanup, fake_embeddings
    ):
        from sqlalchemy import text as sa_text

        from axiom.retrieval.vector_store import get_engine

        filename = f"w0-reingest-{uuid.uuid4().hex[:8]}.txt"
        corpus_cleanup.append(filename)

        r1 = await _ingest(client, filename, b"alpha content for version one. " * 60)
        assert r1.status_code == 200, r1.text
        body1 = r1.json()
        assert body1["vector"] == "indexed"
        old_ids = [c["chunk_id"] for c in body1["chunks"]]
        assert old_ids, "v1 produced no chunks"

        # Enough distinct sentences to produce a different, larger chunk set.
        v2 = (". ".join(f"beta sentence {i} for version two" for i in range(300))) + "."
        r2 = await _ingest(client, filename, v2.encode())
        assert r2.status_code == 200, r2.text
        body2 = r2.json()
        assert body2["vector"] == "indexed"

        engine = get_engine()
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    sa_text("SELECT chunk_id FROM chunk_embeddings WHERE source = :s"),
                    {"s": filename},
                )
            ).fetchall()
        current_ids = {r[0] for r in rows}

        assert old_ids[0] not in current_ids, "stale v1 chunk survived re-ingest"
        assert len(current_ids) == body2["chunk_count"], (
            "chunk rows must match the latest upload exactly — no ON CONFLICT survivors"
        )

    @pytest.mark.asyncio
    async def test_reingest_same_file_is_idempotent(
        self, client, connected_services, corpus_cleanup, fake_embeddings
    ):
        from sqlalchemy import text as sa_text

        from axiom.retrieval.vector_store import get_engine

        filename = f"w0-idempotent-{uuid.uuid4().hex[:8]}.txt"
        corpus_cleanup.append(filename)
        content = b"identical content re-uploaded. " * 60

        r1 = await _ingest(client, filename, content)
        r2 = await _ingest(client, filename, content)
        assert r1.status_code == 200 and r2.status_code == 200

        engine = get_engine()
        async with engine.connect() as conn:
            count = (
                await conn.execute(
                    sa_text("SELECT COUNT(*) FROM chunk_embeddings WHERE source = :s"),
                    {"s": filename},
                )
            ).scalar()
        assert count == r2.json()["chunk_count"]
        assert r1.json()["chunks"][0]["chunk_id"] == r2.json()["chunks"][0]["chunk_id"]


class TestDeleteDocument:
    @pytest.mark.asyncio
    async def test_delete_removes_lineage(
        self, client, connected_services, corpus_cleanup, fake_embeddings
    ):
        from sqlalchemy import text as sa_text

        import server as server_module
        from axiom.retrieval.vector_store import get_engine

        filename = f"w0-delete-{uuid.uuid4().hex[:8]}.txt"
        corpus_cleanup.append(filename)

        r = await _ingest(client, filename, b"delete me content for lineage. " * 60)
        assert r.status_code == 200, r.text
        body = r.json()
        doc_id = body["doc_id"]
        assert doc_id

        # Delete route is protected too.
        assert (
            await client.delete(f"/api/documents/{doc_id}")
        ).status_code == 401

        r_del = await client.delete(f"/api/documents/{doc_id}", headers=AUTH)
        assert r_del.status_code == 200, r_del.text
        data = r_del.json()
        assert data["status"] == "deleted"
        assert data["doc_id"] == doc_id
        assert data["filename"] == filename
        assert data["deleted_chunks"] == body["chunk_count"]

        engine = get_engine()
        async with engine.connect() as conn:
            chunks_left = (
                await conn.execute(
                    sa_text("SELECT COUNT(*) FROM chunk_embeddings WHERE source = :s"),
                    {"s": filename},
                )
            ).scalar()
            docs_left = (
                await conn.execute(
                    sa_text("SELECT COUNT(*) FROM ingested_documents WHERE doc_id = :d"),
                    {"d": doc_id},
                )
            ).scalar()
        assert chunks_left == 0, "chunk embeddings must be removed with the document"
        assert docs_left == 0, "ingested_documents lineage row must be removed"
        assert not any(
            d.get("doc_id") == doc_id for d in server_module._ingested_docs
        ), "in-memory lineage must be removed"

        assert (
            await client.delete(f"/api/documents/{doc_id}", headers=AUTH)
        ).status_code == 404

    @pytest.mark.asyncio
    async def test_delete_unknown_doc_is_404(self, client, connected_services):
        r = await client.delete("/api/documents/deadbeef00000000", headers=AUTH)
        assert r.status_code == 404


class TestCacheInvalidation:
    @pytest.mark.asyncio
    async def test_ingest_clears_semantic_cache(
        self, client, connected_services, corpus_cleanup, fake_embeddings
    ):
        from axiom.cache.semantic_cache import semantic_cache
        from axiom.config import get_config

        if not await semantic_cache.is_connected():
            pytest.skip("Redis not available")

        dims = get_config().embedding_dimensions
        assert await semantic_cache.store(
            "w0 cache probe query",
            [0.01] * dims,
            {"evaluation_passed": True, "final_answer": "cached answer"},
        )
        assert (await semantic_cache.stats())["total_entries"] >= 1

        filename = f"w0-cache-{uuid.uuid4().hex[:8]}.txt"
        corpus_cleanup.append(filename)
        r = await _ingest(client, filename, b"cache invalidation content. " * 60)
        assert r.status_code == 200, r.text

        assert (await semantic_cache.stats())["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_ingest_and_delete_invalidate_cache(
        self, client, fake_embeddings
    ):
        """Spy variant — runs without any external services."""
        import server as server_module

        spy = AsyncMock(return_value=2)
        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(server_module.semantic_cache, "clear", spy)

            filename = f"w0-spy-{uuid.uuid4().hex[:8]}.txt"
            r = await _ingest(client, filename, b"spy cache probe. " * 60)
            assert r.status_code == 200, r.text
            assert spy.await_count == 1

            doc_id = r.json()["doc_id"]
            r_del = await client.delete(f"/api/documents/{doc_id}", headers=AUTH)
            assert r_del.status_code == 200, r_del.text
            assert spy.await_count == 2
        finally:
            monkeypatch.undo()
