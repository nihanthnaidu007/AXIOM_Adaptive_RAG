"""Feedback loop tests (Wave 2, D2).

POST /api/feedback persists a thumbs row against a pipeline trace;
GET /api/feedback/summary aggregates counts and recent items for the eval
dashboard. Persistence + visibility only — automatic strategy re-tuning is
a follow-up wave. Endpoint tests run against real PostgreSQL when available
(skips otherwise, like test_pg_backed_stores.py); the fresh-DB migration
chain itself is proven by test_migrations.py.
"""

import uuid

import pytest
import pytest_asyncio

API_KEY = "test-api-key"  # keep in sync with tests/conftest.py
AUTH = {"X-API-Key": API_KEY}

FEEDBACK_DDL = """
    CREATE TABLE IF NOT EXISTS query_feedback (
        id SERIAL PRIMARY KEY,
        trace_id TEXT NOT NULL REFERENCES pipeline_traces(session_id) ON DELETE CASCADE,
        rating SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
        comment TEXT,
        query_snippet TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
"""


@pytest_asyncio.fixture()
async def feedback_tables():
    """Mirror the pipeline_traces + query_feedback DDL; skip without PostgreSQL."""
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
        await conn.execute(sa_text(FEEDBACK_DDL))
    yield
    from axiom.retrieval.vector_store import get_engine

    engine = get_engine()
    if engine:
        async with engine.begin() as conn:
            await conn.execute(sa_text("DELETE FROM query_feedback"))
            await conn.execute(sa_text("DELETE FROM pipeline_traces"))


async def _insert_trace(session_id: str) -> None:
    from sqlalchemy import text as sa_text

    from axiom.retrieval.vector_store import get_engine

    async with get_engine().begin() as conn:
        await conn.execute(
            sa_text("INSERT INTO pipeline_traces (session_id, trace_data) VALUES (:sid, :data)"),
            {"sid": session_id, "data": "[]"},
        )


@pytest.mark.asyncio
class TestPostFeedback:
    async def test_records_a_thumbs_up_against_an_existing_trace(self, client, feedback_tables):
        trace_id = str(uuid.uuid4())
        await _insert_trace(trace_id)

        response = await client.post(
            "/api/feedback",
            headers=AUTH,
            json={"trace_id": trace_id, "rating": 1, "query_snippet": "what is axiom"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["trace_id"] == trace_id
        assert body["rating"] == 1
        assert body["status"] == "recorded"
        assert body["id"] > 0

    async def test_records_a_thumbs_down_with_optional_comment(self, client, feedback_tables):
        trace_id = str(uuid.uuid4())
        await _insert_trace(trace_id)

        response = await client.post(
            "/api/feedback",
            headers=AUTH,
            json={"trace_id": trace_id, "rating": -1, "comment": "answer missed the point"},
        )

        assert response.status_code == 200
        assert response.json()["rating"] == -1

    async def test_unknown_trace_is_404(self, client, feedback_tables):
        response = await client.post(
            "/api/feedback", headers=AUTH, json={"trace_id": str(uuid.uuid4()), "rating": 1}
        )

        assert response.status_code == 404
        assert "No trace found" in response.json()["detail"]["error"]

    async def test_rating_outside_thumbs_domain_is_422(self, client, feedback_tables):
        response = await client.post(
            "/api/feedback", headers=AUTH, json={"trace_id": str(uuid.uuid4()), "rating": 0}
        )

        assert response.status_code == 422

    async def test_whitespace_comment_is_not_persisted(self, client, feedback_tables):
        from sqlalchemy import text as sa_text

        from axiom.retrieval.vector_store import get_engine

        trace_id = str(uuid.uuid4())
        await _insert_trace(trace_id)

        response = await client.post(
            "/api/feedback", headers=AUTH, json={"trace_id": trace_id, "rating": 1, "comment": "   "}
        )

        assert response.status_code == 200
        async with get_engine().connect() as conn:
            stored = (await conn.execute(
                sa_text("SELECT comment FROM query_feedback WHERE trace_id = :tid"),
                {"tid": trace_id},
            )).fetchone()
        assert stored[0] is None

    async def test_requires_api_key(self, client, feedback_tables):
        response = await client.post(
            "/api/feedback", json={"trace_id": str(uuid.uuid4()), "rating": 1}
        )

        assert response.status_code == 401

    async def test_without_postgres_is_503_sanitized(self, client, monkeypatch):
        # Fail-visible degraded mode: no engine means the write would be lost,
        # so the endpoint refuses instead of pretending to record.
        from axiom.retrieval.vector_store import vector_store

        monkeypatch.setattr(vector_store, "_engine", None)
        monkeypatch.setattr(vector_store, "_connected", False)

        response = await client.post(
            "/api/feedback", headers=AUTH, json={"trace_id": str(uuid.uuid4()), "rating": 1}
        )

        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["error"] == "Feedback storage is not available"


@pytest.mark.asyncio
class TestFeedbackSummary:
    async def test_aggregates_counts_and_recent_items(self, client, feedback_tables):
        trace_id = str(uuid.uuid4())
        await _insert_trace(trace_id)
        await client.post("/api/feedback", headers=AUTH, json={"trace_id": trace_id, "rating": 1})
        await client.post(
            "/api/feedback",
            headers=AUTH,
            json={"trace_id": trace_id, "rating": -1, "comment": "off target", "query_snippet": "q"},
        )

        response = await client.get("/api/feedback/summary", headers=AUTH)

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert body["counts"] == {"up": 1, "down": 1}
        assert len(body["recent"]) == 2
        # Most recent first — the thumbs-down was recorded last.
        assert body["recent"][0]["rating"] == -1
        assert body["recent"][0]["comment"] == "off target"
        assert body["recent"][0]["trace_id"] == trace_id

    async def test_empty_store_returns_zero_counts(self, client, feedback_tables):
        response = await client.get("/api/feedback/summary", headers=AUTH)

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 0
        assert body["counts"] == {"up": 0, "down": 0}
        assert body["recent"] == []

    async def test_degrades_to_zero_counts_without_postgres(self, client, monkeypatch):
        from axiom.retrieval.vector_store import vector_store

        monkeypatch.setattr(vector_store, "_engine", None)
        monkeypatch.setattr(vector_store, "_connected", False)

        response = await client.get("/api/feedback/summary", headers=AUTH)

        assert response.status_code == 200
        assert response.json()["total"] == 0

    async def test_requires_api_key(self, client, feedback_tables):
        response = await client.get("/api/feedback/summary")

        assert response.status_code == 401
