"""Eval results endpoint tests (Wave 5, D4).

GET /api/eval/results serves the newest completed eval run from the
PG-backed eval_runs table (dual-written by the background runner),
reconstructing the historical local-file shape {aggregate, per_query};
the eval_results.json file remains the fallback when PostgreSQL is
unavailable. Endpoint tests run against real PostgreSQL when available
(skips otherwise, like test_eval_runs_endpoint.py); degraded mode reads
the file, and the endpoint 404s when neither source has a completed run.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

import server

API_KEY = "test-api-key"  # keep in sync with tests/conftest.py
AUTH = {"X-API-Key": API_KEY}

# Mirror the FULL post-migration eval_runs schema (initial DDL + the error/
# latest columns added by c4d8e2f6a9b1) — same fixture as
# test_eval_runs_endpoint.py, which may run before or after this module.
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


async def _insert_run(
    job_id: str,
    status: str = "complete",
    aggregate: dict | None = None,
    results: list | None = None,
    started_at: datetime | None = None,
) -> None:
    from sqlalchemy import text as sa_text

    from axiom.retrieval.vector_store import get_engine

    async with get_engine().begin() as conn:
        await conn.execute(
            sa_text(
                "INSERT INTO eval_runs (job_id, status, progress, total, aggregate, results, started_at) "
                "VALUES (:job_id, :status, 5, 5, :aggregate, :results, :started_at)"
            ),
            {
                "job_id": job_id,
                "status": status,
                "aggregate": json.dumps(aggregate or {}),
                "results": json.dumps(results or []),
                "started_at": started_at or datetime(2026, 9, 18, 10, 0, tzinfo=timezone.utc),
            },
        )


@pytest.mark.asyncio
class TestEvalResultsEndpoint:
    async def test_serves_newest_complete_run_from_postgres(
        self, client, eval_runs_table
    ):
        await _insert_run(
            "run-old",
            aggregate={"keyword_hit_rate": 0.5},
            results=[{"session_id": "eval-old"}],
            started_at=datetime(2026, 9, 18, 9, 0, tzinfo=timezone.utc),
        )
        await _insert_run(
            "run-new",
            aggregate={"keyword_hit_rate": 0.9},
            results=[{"session_id": "eval-new"}],
            started_at=datetime(2026, 9, 18, 10, 0, tzinfo=timezone.utc),
        )
        # A newer in-flight run must never shadow the newest completed one.
        await _insert_run(
            "run-running", status="running", started_at=datetime(2026, 9, 18, 11, 0, tzinfo=timezone.utc)
        )

        response = await client.get("/api/eval/results", headers=AUTH)

        assert response.status_code == 200
        body = response.json()
        # The pre-W5 shape is preserved: the file held {aggregate, per_query}.
        assert body["aggregate"]["keyword_hit_rate"] == 0.9
        assert body["per_query"] == [{"session_id": "eval-new"}]

    async def test_404_when_no_completed_run(self, client, eval_runs_table):
        response = await client.get("/api/eval/results", headers=AUTH)

        assert response.status_code == 404
        assert "No eval results" in response.json()["detail"]["error"]

    async def test_404_when_only_incomplete_runs(self, client, eval_runs_table):
        await _insert_run(
            "run-running", status="running", started_at=datetime(2026, 9, 18, 10, 0, tzinfo=timezone.utc)
        )

        response = await client.get("/api/eval/results", headers=AUTH)

        assert response.status_code == 404
        assert "No eval results" in response.json()["detail"]["error"]

    async def test_falls_back_to_local_file_without_postgres(self, client, monkeypatch):
        from axiom.retrieval.vector_store import vector_store

        monkeypatch.setattr(vector_store, "_engine", None)
        monkeypatch.setattr(vector_store, "_connected", False)

        results_path = Path(server.__file__).parent / "eval_results.json"
        payload = {
            "aggregate": {"keyword_hit_rate": 0.73},
            "per_query": [{"session_id": "eval-file"}],
        }
        saved = results_path.read_text() if results_path.exists() else None
        try:
            results_path.write_text(json.dumps(payload))
            response = await client.get("/api/eval/results", headers=AUTH)

            assert response.status_code == 200
            body = response.json()
            assert body["aggregate"]["keyword_hit_rate"] == 0.73
            assert body["per_query"] == [{"session_id": "eval-file"}]
        finally:
            if saved is None:
                results_path.unlink(missing_ok=True)
            else:
                results_path.write_text(saved)

    async def test_requires_api_key(self, client):
        response = await client.get("/api/eval/results")

        assert response.status_code == 401
