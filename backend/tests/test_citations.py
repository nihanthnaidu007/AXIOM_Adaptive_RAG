"""Citations endpoints and trace_data enrichment (Wave 2, D1).

Live citations ride reranked_chunks on the query/SSE payloads (no migration);
historical citations are enriched into pipeline_traces.trace_data at write
time and served back by GET /api/citations/{trace_id}. The enrichment is a
pure list-mutation — covered here directly — and the endpoint is covered
against a real PostgreSQL when available (skips otherwise, like
test_pg_backed_stores.py).
"""

import uuid

import pytest
import pytest_asyncio

from server import (
    MAX_PERSISTED_CITATIONS,
    _attach_trace_citations,
    _citations_from_state,
    _extract_trace_citations,
)

API_KEY = "test-api-key"  # keep in sync with tests/conftest.py
AUTH = {"X-API-Key": API_KEY}


def _chunk(i: int, content: str = "chunk body") -> dict:
    return {
        "chunk_id": f"chunk-{i}",
        "content": content,
        "source": "handbook.pdf",
        "bm25_score": 1.0,
        "vector_score": 0.5,
        "rrf_score": 0.03,
        "rerank_score": 0.9 - i * 0.1,
        "pre_rerank_position": i + 1,
        "post_rerank_position": i,
    }


def _rerank_step() -> dict:
    return {"node_name": "rerank_chunks", "status": "complete", "summary": "ok"}


class TestCitationsFromState:
    def test_projects_citation_fields_and_excerpts_content(self):
        state = {"reranked_chunks": [_chunk(0, content="x" * 1000)]}

        citations = _citations_from_state(state)

        assert len(citations) == 1
        c = citations[0]
        assert c["chunk_id"] == "chunk-0"
        assert c["source"] == "handbook.pdf"
        assert len(c["content"]) == 600  # excerpted, not the full 1000 chars
        assert c["rerank_score"] == 0.9
        assert c["pre_rerank_position"] == 1
        assert c["post_rerank_position"] == 0

    def test_caps_at_the_persisted_citation_limit(self):
        state = {"reranked_chunks": [_chunk(i) for i in range(MAX_PERSISTED_CITATIONS + 5)]}

        assert len(_citations_from_state(state)) == MAX_PERSISTED_CITATIONS

    def test_empty_state_yields_no_citations(self):
        assert _citations_from_state({}) == []
        assert _citations_from_state({"reranked_chunks": []}) == []


class TestAttachTraceCitations:
    def test_enriches_the_rerank_step_detail_in_place(self):
        steps = [_rerank_step(), {"node_name": "generate_answer", "status": "complete", "summary": "ok"}]
        citations = [_chunk(0)]

        _attach_trace_citations(steps, citations)

        assert steps[0]["detail"]["citations"] == citations
        assert "detail" not in steps[1]
        # /trace consumers still see a list of plain trace steps.
        assert [s["node_name"] for s in steps] == ["rerank_chunks", "generate_answer"]

    def test_falls_back_to_the_check_cache_step_on_cache_hits(self):
        steps = [{"node_name": "check_cache", "status": "complete", "summary": "hit"}]

        _attach_trace_citations(steps, [_chunk(0)])

        assert steps[0]["detail"]["citations"][0]["chunk_id"] == "chunk-0"

    def test_no_rerank_step_is_a_no_op(self):
        steps = [{"node_name": "generate_answer", "status": "complete", "summary": "ok"}]

        _attach_trace_citations(steps, [_chunk(0)])

        assert "detail" not in steps[0]

    def test_no_citations_is_a_no_op(self):
        steps = [_rerank_step()]

        _attach_trace_citations(steps, [])

        assert "detail" not in steps[0]


class TestExtractTraceCitations:
    def test_round_trips_through_attach(self):
        steps = [_rerank_step(), {"node_name": "generate_answer", "status": "complete", "summary": "ok"}]
        citations = [_chunk(0), _chunk(1)]
        _attach_trace_citations(steps, citations)

        assert _extract_trace_citations(steps) == citations

    def test_legacy_traces_without_citations_extract_empty(self):
        steps = [_rerank_step()]

        assert _extract_trace_citations(steps) == []

    def test_non_list_trace_data_extracts_empty(self):
        assert _extract_trace_citations({"unexpected": "shape"}) == []


@pytest_asyncio.fixture()
async def traces_table():
    """Mirror the pipeline_traces DDL; skip when PostgreSQL is unavailable."""
    from sqlalchemy import text as sa_text

    from axiom.retrieval.vector_store import vector_store

    if not await vector_store.connect():
        pytest.skip("PostgreSQL/pgvector not available — store tests require it")
    async with vector_store._engine.begin() as conn:  # type: ignore[union-attr]
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS pipeline_traces (
                session_id TEXT PRIMARY KEY,
                trace_data JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
    yield
    from axiom.retrieval.vector_store import get_engine

    engine = get_engine()
    if engine:
        async with engine.begin() as conn:
            await conn.execute(sa_text("DELETE FROM pipeline_traces"))


async def _insert_trace(session_id: str, trace_data: list) -> None:
    import json

    from sqlalchemy import text as sa_text

    from axiom.retrieval.vector_store import get_engine

    async with get_engine().begin() as conn:
        await conn.execute(
            sa_text(
                "INSERT INTO pipeline_traces (session_id, trace_data) VALUES (:sid, :data)"
            ),
            {"sid": session_id, "data": json.dumps(trace_data)},
        )


@pytest.mark.asyncio
class TestCitationsEndpoint:
    async def test_serves_citations_enriched_into_a_historical_trace(self, client, traces_table):
        import json

        session_id = str(uuid.uuid4())
        steps = [_rerank_step(), {"node_name": "generate_answer", "status": "complete", "summary": "ok"}]
        _attach_trace_citations(steps, [_chunk(0), _chunk(1)])
        await _insert_trace(session_id, json.loads(json.dumps(steps)))

        response = await client.get(f"/api/citations/{session_id}", headers=AUTH)

        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == session_id
        assert [c["chunk_id"] for c in body["citations"]] == ["chunk-0", "chunk-1"]
        assert body["citations"][0]["source"] == "handbook.pdf"

    async def test_existing_trace_without_enrichment_returns_empty_citations(self, client, traces_table):
        import json

        session_id = str(uuid.uuid4())
        await _insert_trace(session_id, json.loads(json.dumps([_rerank_step()])))

        response = await client.get(f"/api/citations/{session_id}", headers=AUTH)

        assert response.status_code == 200
        assert response.json()["citations"] == []

    async def test_missing_trace_is_404(self, client, traces_table):
        response = await client.get(f"/api/citations/{uuid.uuid4()}", headers=AUTH)

        assert response.status_code == 404
        assert "No trace found" in response.json()["detail"]["error"]

    async def test_unknown_trace_404_without_postgres(self, client, monkeypatch):
        # Degraded mode: no engine, empty in-memory store — the endpoint still
        # answers 404 instead of raising.
        from axiom.retrieval.vector_store import vector_store

        monkeypatch.setattr(vector_store, "_engine", None)
        monkeypatch.setattr(vector_store, "_connected", False)

        response = await client.get(f"/api/citations/{uuid.uuid4()}", headers=AUTH)

        assert response.status_code == 404

    async def test_requires_api_key(self, client, traces_table):
        response = await client.get(f"/api/citations/{uuid.uuid4()}")

        assert response.status_code == 401
