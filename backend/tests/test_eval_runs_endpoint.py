"""Eval runs endpoint tests (Wave 2, D3).

GET /api/eval/runs lists the PG-backed eval_runs table (newest first) for
the /eval dashboard. Endpoint tests run against real PostgreSQL when
available (skips otherwise, like test_pg_backed_stores.py); degraded mode
returns an empty list rather than failing the dashboard.
"""


import pytest
import pytest_asyncio

API_KEY = "test-api-key"  # keep in sync with tests/conftest.py
AUTH = {"X-API-Key": API_KEY}

# Mirror the FULL post-migration eval_runs schema (initial DDL + the error/
# latest columns added by c4d8e2f6a9b1). This fixture may run BEFORE
# test_pg_backed_stores.py's eval-job tests, so the table it creates must
# already carry every column the eval-job upsert path writes.
EVAL_RUNS_DDL = """
    CREATE TABLE IF NOT EXISTS eval_runs (
        job_id TEXT PRIMARY KEY,
        status TEXT,
        progress INTEGER,
        total INTEGER,
        aggregate JSONB,
        results JSONB,
        started_at TIMESTAMPTZ DEFAULT NOW(),
        completed_at TIMESTAMPTZ,
        error TEXT,
        latest JSONB
    )
"""


@pytest_asyncio.fixture()
async def eval_runs_table():
    """Mirror the eval_runs DDL; skip when PostgreSQL is unavailable."""
    from sqlalchemy import text as sa_text

    from axiom.retrieval.vector_store import vector_store

    if not await vector_store.connect():
        pytest.skip("PostgreSQL/pgvector not available — store tests require it")
    async with vector_store._engine.begin() as conn:  # type: ignore[union-attr]
        await conn.execute(sa_text(EVAL_RUNS_DDL))
    yield
    from axiom.retrieval.vector_store import get_engine

    engine = get_engine()
    if engine:
        async with engine.begin() as conn:
            await conn.execute(sa_text("DELETE FROM eval_runs"))


async def _insert_run(job_id: str, status: str = "complete", aggregate: dict | None = None) -> None:
    import json

    from sqlalchemy import text as sa_text

    from axiom.retrieval.vector_store import get_engine

    async with get_engine().begin() as conn:
        await conn.execute(
            sa_text(
                "INSERT INTO eval_runs (job_id, status, progress, total, aggregate) "
                "VALUES (:job_id, :status, 5, 5, :aggregate)"
            ),
            {
                "job_id": job_id,
                "status": status,
                "aggregate": json.dumps(aggregate or {"keyword_hit_rate": 0.73}),
            },
        )


@pytest.mark.asyncio
class TestEvalRunsEndpoint:
    async def test_lists_persisted_runs_newest_first(self, client, eval_runs_table):
        await _insert_run("run-old", aggregate={"keyword_hit_rate": 0.5})
        await _insert_run("run-new", aggregate={"keyword_hit_rate": 0.9})

        response = await client.get("/api/eval/runs", headers=AUTH)

        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 2
        assert [r["job_id"] for r in body["runs"]] == ["run-new", "run-old"]
        assert body["runs"][0]["aggregate"]["keyword_hit_rate"] == 0.9
        assert body["runs"][0]["status"] == "complete"

    async def test_empty_table_returns_empty_list(self, client, eval_runs_table):
        response = await client.get("/api/eval/runs", headers=AUTH)

        assert response.status_code == 200
        assert response.json() == {"runs": [], "count": 0}

    async def test_degrades_to_empty_list_without_postgres(self, client, monkeypatch):
        from axiom.retrieval.vector_store import vector_store

        monkeypatch.setattr(vector_store, "_engine", None)
        monkeypatch.setattr(vector_store, "_connected", False)

        response = await client.get("/api/eval/runs", headers=AUTH)

        assert response.status_code == 200
        assert response.json() == {"runs": [], "count": 0}

    async def test_requires_api_key(self, client, eval_runs_table):
        response = await client.get("/api/eval/runs")

        assert response.status_code == 401
