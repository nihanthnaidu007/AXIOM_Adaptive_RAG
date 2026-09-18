"""S3 connector integration tests — run against real MinIO in CI.

The `s3-connector-integration` CI job starts a MinIO container, creates the
`axiom-test` bucket, and uploads `tests/fixtures/pdf/quarterly_report_table.pdf`
as `reports/quarterly_report_table.pdf`. These tests then exercise the REAL
wire path (boto3 → MinIO) and the shared parse seam on real fixture bytes.

Everything downstream of parsing (indexing, cache, persistence) is stubbed:
integration here means "S3 wire format + fetch + parse", not model APIs.

Each test skips itself unless S3_BUCKET is configured, so local runs without
MinIO stay green.
"""

import pytest
import pytest_asyncio

API_KEY = "test-api-key"  # keep in sync with tests/conftest.py and the CI job
AUTH = {"X-API-Key": API_KEY}

EXPECTED_OBJECT_KEY = "reports/quarterly_report_table.pdf"


def _skip_unless_configured():
    from axiom.connectors.s3_connector import is_s3_configured

    if not is_s3_configured():
        pytest.skip(
            "S3 integration requires S3_BUCKET (and a running MinIO); "
            "the s3-connector-integration CI job provides it"
        )


@pytest_asyncio.fixture()
async def client():
    from httpx import ASGITransport, AsyncClient

    from server import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture()
def stubbed_pipeline(monkeypatch):
    """Stub indexing/cache/persistence; record calls for boundary assertions."""
    import server

    indexer_calls = []

    class _StubIndexer:
        async def index_run(self, chunks):
            indexer_calls.append(list(chunks))
            return {"vector": "indexed", "bm25": "indexed", "chunk_count": len(chunks)}

    cleared = {"count": 0}

    async def _fake_clear():
        cleared["count"] += 1
        return 1

    persisted = []

    async def _fake_persist(**kwargs):
        persisted.append(kwargs)

    monkeypatch.setattr(server, "get_dual_indexer", lambda: _StubIndexer())
    monkeypatch.setattr(server.semantic_cache, "clear", _fake_clear)
    monkeypatch.setattr(server, "_persist_ingested_doc", _fake_persist)
    return {"indexer_calls": indexer_calls, "cleared": cleared, "persisted": persisted}


@pytest.mark.asyncio
async def test_fetch_s3_objects_over_real_wire():
    """boto3 lists and fetches the uploaded fixture from MinIO."""
    from axiom.connectors.s3_connector import fetch_s3_objects

    _skip_unless_configured()

    objects, errors = await fetch_s3_objects()

    assert errors == []
    sources = [o.source for o in objects]
    assert f"s3://axiom-test/{EXPECTED_OBJECT_KEY}" in sources

    obj = next(o for o in objects if o.source == f"s3://axiom-test/{EXPECTED_OBJECT_KEY}")
    assert obj.filename == "quarterly_report_table.pdf"
    assert obj.content.startswith(b"%PDF-"), "real PDF bytes must arrive unmodified"
    assert obj.size == len(obj.content)
    assert obj.content_hash
    assert obj.fetched_at is not None


@pytest.mark.asyncio
async def test_fetched_fixture_parses_with_page_provenance():
    """The real fixture bytes flow through the shared parse seam."""
    from axiom.ingest.extract import parse_document

    _skip_unless_configured()

    from axiom.connectors.s3_connector import fetch_s3_objects

    objects, errors = await fetch_s3_objects()
    assert errors == []
    obj = next(o for o in objects if o.source == f"s3://axiom-test/{EXPECTED_OBJECT_KEY}")

    outcome = parse_document(
        obj.content,
        ".pdf",
        ocr_enabled=False,  # digital fixture: pdfplumber path, no OCR
    )

    assert outcome.pages, "digital fixture must yield at least one page"
    assert all(p["page_num"] >= 1 for p in outcome.pages)
    text = "".join(p["text"] for p in outcome.pages)
    assert "revenue" in text.lower(), "fixture text layer carries the financial table"
    assert outcome.parse_confidence == 1.0


@pytest.mark.asyncio
async def test_full_s3_connector_run_against_minio(
    client, stubbed_pipeline
):
    """End to end: authenticated run start → real MinIO fetch → shared parse
    seam → run-level batch → terminal completed status."""

    _skip_unless_configured()

    start = await client.post("/api/connectors/s3/run", headers=AUTH)
    assert start.status_code == 202
    run_id = start.json()["run_id"]

    # ASGITransport runs BackgroundTasks before the POST call returns.
    poll = await client.get(f"/api/connectors/runs/{run_id}", headers=AUTH)
    assert poll.status_code == 200
    run = poll.json()

    assert run["connector"] == "s3"
    assert run["status"] == "completed", f"run errors: {run['errors']}"
    assert run["completed_at"] is not None

    doc = next(
        d for d in run["documents"] if d["source"] == f"s3://axiom-test/{EXPECTED_OBJECT_KEY}"
    )
    assert doc["status"] == "completed"
    assert doc["origin_type"] == "s3"
    assert doc["origin_uri"] == f"s3://axiom-test/{EXPECTED_OBJECT_KEY}"
    assert doc["chunk_count"] > 0
    assert doc["parse_confidence"] == 1.0

    # Batch boundary: ONE index_run call for the whole run, ONE cache clear.
    assert len(stubbed_pipeline["indexer_calls"]) == 1
    assert stubbed_pipeline["cleared"]["count"] == 1

    persisted = stubbed_pipeline["persisted"][0]
    assert persisted["origin_type"] == "s3"
    assert persisted["status"] == "completed"
    assert persisted["parse_confidence"] == 1.0
