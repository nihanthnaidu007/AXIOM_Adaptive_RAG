"""Postgres-backed trace and eval-job store tests (Wave 2).

Before W2, trace reads preferred the in-process store and eval-job state
lived only in a per-process dict — the app could not run more than one
worker with coherent state. These tests prove the Postgres tables are the
source of truth: rows inserted into ``pipeline_traces`` / ``eval_runs`` are
served by the API even when the in-process stores are empty, which is the
state any second worker is in.

Connects the store singletons directly (no app lifespan — that would load
models and connect the checkpointer); skips when PostgreSQL is unavailable —
the CI workflow provisions it. Mirrors the initial-migration DDL (plus the
W2 eval_runs columns) so fresh CI databases have the schema under test.
"""

import uuid

import pytest
import pytest_asyncio

API_KEY = "test-api-key"  # keep in sync with tests/conftest.py
AUTH = {"X-API-Key": API_KEY}


@pytest_asyncio.fixture()
async def connected_services():
    """Connect store singletons and mirror the migration DDL for the tables
    under test (vector_store.connect() self-creates only chunk_embeddings)."""
    from sqlalchemy import text as sa_text

    from axiom.cache.semantic_cache import semantic_cache
    from axiom.retrieval.vector_store import vector_store

    if not await vector_store.connect():
        pytest.skip("PostgreSQL/pgvector not available — store tests require it")
    await semantic_cache.connect()
    async with vector_store._engine.begin() as conn:  # type: ignore[union-attr]
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS pipeline_traces (
                session_id TEXT PRIMARY KEY,
                trace_data JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        await conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                job_id TEXT PRIMARY KEY,
                status TEXT,
                progress INTEGER,
                total INTEGER,
                aggregate JSONB,
                results JSONB,
                error TEXT,
                latest JSONB,
                started_at TIMESTAMPTZ DEFAULT NOW(),
                completed_at TIMESTAMPTZ
            )
        """))
    yield
    from axiom.retrieval.vector_store import get_engine

    engine = get_engine()
    if engine:
        async with engine.begin() as conn:
            await conn.execute(sa_text("DELETE FROM pipeline_traces"))
            await conn.execute(sa_text("DELETE FROM eval_runs"))


@pytest_asyncio.fixture()
async def pg_cleanup(connected_services):
    """Track inserted rows and delete them after the test."""
    from sqlalchemy import text as sa_text

    from axiom.retrieval.vector_store import get_engine

    inserted: list[tuple[str, str]] = []  # (table, key)

    async def _insert(table: str, key: str, sql: str, params: dict):
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.execute(sa_text(sql), params)
        inserted.append((table, key))

    yield _insert

    engine = get_engine()
    if engine:
        async with engine.begin() as conn:
            for table, key in inserted:
                key_col = "session_id" if table == "pipeline_traces" else "job_id"
                await conn.execute(
                    sa_text(f"DELETE FROM {table} WHERE {key_col} = :k"), {"k": key}
                )


def _trace_json(steps: list) -> str:
    """Trace data as a JSON string (how asyncpg receives JSONB)."""
    import json

    return json.dumps(steps)


class TestTraceStorePostgres:
    @pytest.mark.asyncio
    async def test_trace_endpoint_reads_postgres(self, client, pg_cleanup):
        """A trace that exists only in Postgres is served by /api/trace —
        the state any non-writer worker finds itself in."""
        import server as server_module

        session_id = str(uuid.uuid4())
        trace_steps = [
            {"node_name": "classify_query", "status": "ok", "summary": "classified"},
            {"node_name": "generate", "status": "ok", "summary": "generated"},
        ]
        await pg_cleanup(
            "pipeline_traces",
            session_id,
            "INSERT INTO pipeline_traces (session_id, trace_data) VALUES (:sid, :td)",
            {"sid": session_id, "td": _trace_json(trace_steps)},
        )
        # The in-process store must not satisfy the read.
        server_module._trace_store.pop(session_id, None)

        r = await client.get(f"/api/trace/{session_id}", headers=AUTH)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["session_id"] == session_id
        assert [s["node_name"] for s in body["trace_steps"]] == [
            "classify_query",
            "generate",
        ]

    @pytest.mark.asyncio
    async def test_jsonb_string_decoded(self, client, pg_cleanup):
        """asyncpg hands JSONB back as a raw string; the loader must decode
        it to the list the response model promises."""
        from sqlalchemy import text as sa_text

        import server as server_module
        from axiom.retrieval.vector_store import get_engine

        session_id = str(uuid.uuid4())
        steps = [{"node_name": "generate", "status": "ok", "summary": "g"}]
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.execute(
                sa_text(
                    "INSERT INTO pipeline_traces (session_id, trace_data) "
                    "VALUES (:sid, :td)"
                ),
                {"sid": session_id, "td": _trace_json(steps)},
            )
        server_module._trace_store.pop(session_id, None)

        loaded = await server_module._load_trace(session_id)
        assert loaded == steps, f"JSONB not decoded: {loaded!r}"


class TestEvalJobsPostgres:
    @pytest.mark.asyncio
    async def test_eval_status_reads_postgres(self, client, pg_cleanup):
        """An eval job tracked only in Postgres is served by /eval/status —
        the caller no longer needs the worker that started the run."""
        import server as server_module

        job_id = uuid.uuid4().hex[:12]
        await pg_cleanup(
            "eval_runs",
            job_id,
            """INSERT INTO eval_runs (job_id, status, progress, total, aggregate, results, error, latest)
               VALUES (:jid, :st, :pr, :tot, :agg, :res, :err, :lat)""",
            {
                "jid": job_id,
                "st": "running",
                "pr": 3,
                "tot": 30,
                "agg": None,
                "res": "[]",
                "err": None,
                "lat": '{"query": "q", "latency_s": 1.2}',
            },
        )
        server_module._eval_jobs.pop(job_id, None)

        r = await client.get(f"/api/eval/status/{job_id}", headers=AUTH)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["job_id"] == job_id
        assert body["status"] == "running"
        assert body["progress"] == 3
        assert body["total"] == 30
        assert body["latest"] == {"query": "q", "latency_s": 1.2}

    @pytest.mark.asyncio
    async def test_upsert_eval_run_persists_progress(self, connected_services):
        """_upsert_eval_run writes the full status row, timestamps included."""
        from sqlalchemy import text as sa_text

        import server as server_module
        from axiom.retrieval.vector_store import get_engine

        job_id = uuid.uuid4().hex[:12]
        job = {
            "status": "running",
            "progress": 2,
            "total": 30,
            "results": [],
            "latest": {"query": "q2"},
            "aggregate": None,
            "error": None,
            "started_at": "2026-09-17T10:00:00+00:00",
        }
        try:
            await server_module._upsert_eval_run(job_id, job)
            engine = get_engine()
            async with engine.connect() as conn:
                row = (
                    await conn.execute(
                        sa_text(
                            "SELECT status, progress, latest, started_at FROM eval_runs "
                            "WHERE job_id = :jid"
                        ),
                        {"jid": job_id},
                    )
                ).fetchone()
            assert row is not None, "eval job row missing after upsert"
            assert row[0] == "running"
            assert row[1] == 2
            assert row[3] is not None, "started_at not persisted"

            # Progress tick updates in place (no duplicate row).
            job["progress"] = 3
            await server_module._upsert_eval_run(job_id, job)
            async with engine.connect() as conn:
                count = (
                    await conn.execute(
                        sa_text("SELECT COUNT(*) FROM eval_runs WHERE job_id = :jid"),
                        {"jid": job_id},
                    )
                ).scalar()
            assert count == 1
        finally:
            engine = get_engine()
            async with engine.begin() as conn:
                await conn.execute(
                    sa_text("DELETE FROM eval_runs WHERE job_id = :jid"), {"jid": job_id}
                )

    @pytest.mark.asyncio
    async def test_unknown_job_is_404(self, client, connected_services):
        r = await client.get("/api/eval/status/no-such-job", headers=AUTH)
        assert r.status_code == 404
        assert r.json()["detail"]["error"] == "Job not found"


class TestStatsPostgres:
    @pytest.mark.asyncio
    async def test_stats_uses_postgres_counts(self, client, connected_services):
        """Stats counts come from Postgres (all workers agree), not from
        process-local dict sizes."""
        from sqlalchemy import text as sa_text

        from axiom.retrieval.vector_store import get_engine

        engine = get_engine()
        async with engine.connect() as conn:
            expected_docs = (
                await conn.execute(sa_text("SELECT COUNT(*) FROM ingested_documents"))
            ).scalar()

        r = await client.get("/api/stats", headers=AUTH)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["indexed_documents"] == expected_docs
        assert "total_queries_processed" in body
