"""Wave 4 connector-run lifecycle tests (D3): background ingestion + polling.

Covers:
- authenticated start (202 + pending run id) and authenticated polling endpoint
- fail-closed auth: 401 on missing/wrong key for both endpoints
- 503 CONNECTOR_NOT_CONFIGURED for unconfigured connectors, 404 for unknown ones
- background executor terminal states: completed / partial / failed
- exactly one index_run batch and one cache clear per run
- per-document failure rows carry sanitized reasons and never kill the run
- bookkeeping persistence per document (status lifecycle, provenance)
"""

from datetime import datetime, timezone

import pytest
import pytest_asyncio

API_KEY = "test-api-key"  # keep in sync with tests/conftest.py
AUTH = {"X-API-Key": API_KEY}

LONG_TEXT = (
    "The adaptive retrieval pipeline processes every connector document through "
    "the same parse seam. " * 30
)


def _make_item(source: str, content: bytes | None = None, ext: str = ".txt"):
    body = content if content is not None else LONG_TEXT.encode("utf-8")
    return {
        "source": source,
        "filename": source.rsplit("/", 1)[-1],
        "content": body,
        "ext": ext,
        "fetched_at": datetime.now(timezone.utc),
        "content_hash": "a" * 64,
        "size": len(body),
    }


@pytest.fixture()
def clean_runs():
    """Isolate the module-level run store per test."""
    import server

    server._connector_runs.clear()
    yield
    server._connector_runs.clear()


class _StubIndexer:
    """Records index_run calls; returns a configurable result."""

    def __init__(self, result=None):
        self.result = result or {"vector": "indexed", "bm25": "indexed", "chunk_count": 3}
        self.calls = []

    async def index_run(self, chunks):
        self.calls.append(list(chunks))
        return dict(self.result)


@pytest.fixture()
def stubbed_pipeline(monkeypatch):
    """Stub the indexing + cache + persistence seams around the executor."""
    import server

    indexer = _StubIndexer()
    cleared = {"count": 0}

    async def _fake_clear():
        cleared["count"] += 1
        return 5

    persisted = []

    async def _fake_persist(**kwargs):
        persisted.append(kwargs)

    monkeypatch.setattr(server, "get_dual_indexer", lambda: indexer)
    monkeypatch.setattr(server.semantic_cache, "clear", _fake_clear)
    monkeypatch.setattr(server, "_persist_ingested_doc", _fake_persist)
    return {"indexer": indexer, "cleared": cleared, "persisted": persisted}


@pytest_asyncio.fixture()
async def client():
    from httpx import ASGITransport, AsyncClient

    from server import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_start_unknown_connector_404(client, clean_runs):
    resp = await client.post("/api/connectors/webhook/run", headers=AUTH)

    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "CONNECTOR_NOT_FOUND"


@pytest.mark.asyncio
async def test_start_unconfigured_s3_503(client, clean_runs, monkeypatch):
    import server

    monkeypatch.setattr(server, "is_s3_configured", lambda: False)

    resp = await client.post("/api/connectors/s3/run", headers=AUTH)

    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "CONNECTOR_NOT_CONFIGURED"


@pytest.mark.asyncio
async def test_start_unconfigured_crawl_503(client, clean_runs, monkeypatch):
    import server

    monkeypatch.setattr(server, "is_crawl_configured", lambda: False)

    resp = await client.post("/api/connectors/crawl/run", headers=AUTH)

    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "CONNECTOR_NOT_CONFIGURED"


@pytest.mark.asyncio
async def test_start_and_poll_require_api_key(client, clean_runs):
    start = await client.post("/api/connectors/s3/run")
    poll = await client.get("/api/connectors/runs/whatever")

    assert start.status_code == 401
    assert poll.status_code == 401


@pytest.mark.asyncio
async def test_poll_unknown_run_404(client, clean_runs):
    resp = await client.get("/api/connectors/runs/nope", headers=AUTH)

    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "RUN_NOT_FOUND"


@pytest.mark.asyncio
async def test_happy_path_run_completes_with_single_batch_and_cache_clear(
    client, clean_runs, stubbed_pipeline, monkeypatch
):
    """crawl run: one doc ingested -> completed; ONE index batch, ONE cache clear."""
    import server

    async def _fake_fetch(connector):
        return [_make_item("https://example.com/report")], []

    monkeypatch.setattr(server, "is_crawl_configured", lambda: True)
    monkeypatch.setattr(server, "_fetch_connector_documents", _fake_fetch)

    start = await client.post("/api/connectors/crawl/run", headers=AUTH)
    assert start.status_code == 202
    body = start.json()
    assert body["status"] == "pending"
    run_id = body["run_id"]

    # ASGITransport executes BackgroundTasks before the POST call returns.
    poll = await client.get(f"/api/connectors/runs/{run_id}", headers=AUTH)
    assert poll.status_code == 200
    run = poll.json()

    assert run["status"] == "completed"
    assert run["completed_at"] is not None
    assert run["cache_cleared"] is True
    assert run["indexed_chunks"] == 3
    assert len(run["documents"]) == 1
    doc = run["documents"][0]
    assert doc["status"] == "completed"
    assert doc["source"] == "https://example.com/report"
    assert doc["origin_type"] == "crawl"
    assert doc["origin_uri"] == "https://example.com/report"
    assert doc["chunk_count"] > 0
    assert doc["parse_confidence"] == 1.0

    # Batch-boundary contract: ONE index_run call with all chunks, ONE clear.
    assert len(stubbed_pipeline["indexer"].calls) == 1
    assert stubbed_pipeline["cleared"]["count"] == 1
    # Bookkeeping persisted once for the document.
    assert len(stubbed_pipeline["persisted"]) == 1
    persisted = stubbed_pipeline["persisted"][0]
    assert persisted["origin_type"] == "crawl"
    assert persisted["origin_uri"] == "https://example.com/report"
    assert persisted["status"] == "completed"


@pytest.mark.asyncio
async def test_partial_run_when_one_document_yields_nothing(
    client, clean_runs, stubbed_pipeline, monkeypatch
):
    import server

    async def _fake_fetch(connector):
        return [
            _make_item("https://example.com/good"),
            _make_item("https://example.com/empty", content=b"   "),
        ], []

    monkeypatch.setattr(server, "is_crawl_configured", lambda: True)
    monkeypatch.setattr(server, "_fetch_connector_documents", _fake_fetch)

    start = await client.post("/api/connectors/crawl/run", headers=AUTH)
    run_id = start.json()["run_id"]
    run = (await client.get(f"/api/connectors/runs/{run_id}", headers=AUTH)).json()

    assert run["status"] == "partial"
    by_source = {d["source"]: d for d in run["documents"]}
    assert by_source["https://example.com/good"]["status"] == "completed"
    assert by_source["https://example.com/empty"]["status"] == "failed"
    assert "No indexable content" in by_source["https://example.com/empty"]["error"]


@pytest.mark.asyncio
async def test_failed_run_when_every_document_fails(
    client, clean_runs, stubbed_pipeline, monkeypatch
):
    import server

    async def _fake_fetch(connector):
        return [_make_item("https://example.com/empty", content=b"")], []

    monkeypatch.setattr(server, "is_crawl_configured", lambda: True)
    monkeypatch.setattr(server, "_fetch_connector_documents", _fake_fetch)

    start = await client.post("/api/connectors/crawl/run", headers=AUTH)
    run_id = start.json()["run_id"]
    run = (await client.get(f"/api/connectors/runs/{run_id}", headers=AUTH)).json()

    assert run["status"] == "failed"
    assert run["documents"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_indexing_failure_is_recorded_not_silent(
    client, clean_runs, monkeypatch
):
    """A vector_error result must surface in run errors — no empty-but-successful.

    The cache still clears: BM25 reindexed, so the index contents changed and
    cached answers may reference stale retrieval state either way.
    """
    import server

    indexer = _StubIndexer(
        result={"vector": "error", "bm25": "indexed", "chunk_count": 3, "vector_error": "dimension mismatch"}
    )

    async def _fake_clear():
        return 0

    async def _fake_persist(**kwargs):
        return None

    monkeypatch.setattr(server, "get_dual_indexer", lambda: indexer)
    monkeypatch.setattr(server.semantic_cache, "clear", _fake_clear)
    monkeypatch.setattr(server, "_persist_ingested_doc", _fake_persist)

    async def _fake_fetch(connector):
        return [_make_item("https://example.com/report")], []

    monkeypatch.setattr(server, "is_crawl_configured", lambda: True)
    monkeypatch.setattr(server, "_fetch_connector_documents", _fake_fetch)

    start = await client.post("/api/connectors/crawl/run", headers=AUTH)
    run_id = start.json()["run_id"]
    run = (await client.get(f"/api/connectors/runs/{run_id}", headers=AUTH)).json()

    assert run["cache_cleared"] is True  # index contents changed -> stale answers must go
    assert any(e["source"] == "indexing" and "dimension mismatch" in e["reason"] for e in run["errors"])
    assert run["status"] == "partial"  # the doc parsed fine; vector indexing failed


@pytest.mark.asyncio
async def test_s3_bucket_failure_fails_the_run(client, clean_runs, monkeypatch):
    """Bucket-level listing failure -> run failed with a sanitized reason row."""
    import server
    from axiom.connectors.s3_connector import S3ConnectorError

    async def _fake_fetch(connector):
        raise S3ConnectorError("S3 listing failed for bucket 'axiom-test': denied")

    monkeypatch.setattr(server, "is_s3_configured", lambda: True)
    monkeypatch.setattr(server, "_fetch_connector_documents", _fake_fetch)

    start = await client.post("/api/connectors/s3/run", headers=AUTH)
    run_id = start.json()["run_id"]
    run = (await client.get(f"/api/connectors/runs/{run_id}", headers=AUTH)).json()

    assert run["status"] == "failed"
    assert run["errors"][0]["source"] == "s3"
    assert "listing failed" in run["errors"][0]["reason"]


@pytest.mark.asyncio
async def test_run_record_survives_polling_and_includes_config(client, clean_runs, monkeypatch):
    import server

    monkeypatch.setattr(server, "is_s3_configured", lambda: True)

    async def _never_called(connector):
        raise AssertionError("no fetch expected in this test")

    monkeypatch.setattr(server, "_fetch_connector_documents", _never_called)

    start = await client.post("/api/connectors/s3/run", headers=AUTH)
    run_id = start.json()["run_id"]

    # The executor runs async; after the POST the run exists with a config summary.
    poll = await client.get(f"/api/connectors/runs/{run_id}", headers=AUTH)
    run = poll.json()

    assert run["run_id"] == run_id
    assert run["connector"] == "s3"
    assert "bucket" in run["config"]  # s3_run_config_summary shape
    assert run["started_at"] is not None
