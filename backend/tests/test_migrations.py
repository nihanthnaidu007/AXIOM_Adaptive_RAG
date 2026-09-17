"""Fresh-database migration verification (alembic is the schema single source).

Proves that `alembic upgrade head` alone builds the full application schema on
an empty database, that it downgrades cleanly, and that databases created by
the legacy startup path (tables exist, no alembic_version) upgrade in place.

Creates throwaway scratch databases; the base test user is a superuser in both
the CI container and local setups. Skips when PostgreSQL is unavailable — the
CI workflow provisions it. The alembic subprocess runs sync (blocking the test
loop is fine here); assertions run on the test's own event loop.
"""

import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import pytest_asyncio

BACKEND_DIR = Path(__file__).resolve().parent.parent

EXPECTED_TABLES = {
    "chunk_embeddings",
    "pipeline_traces",
    "ingested_documents",
    "eval_runs",
    "alembic_version",
}


def _base_database_url() -> str:
    """Base URL under which scratch databases are created.

    Deliberately NOT DATABASE_URL: the repo-root .env (loaded by dotenv when
    the app config imports mid-test-run) points at a developer's local stack
    (user `axiom`) that does not exist where these tests run. The default
    matches the CI service container (pgvector/pgvector:pg16).
    """
    return os.environ.get(
        "AXIOM_MIGRATION_TEST_URL",
        "postgresql+asyncpg://axiom_test:axiom_test@localhost:5432/axiom_test",
    )


def _dsn(url: str) -> str:
    """SQLAlchemy engine URL → plain asyncpg DSN."""
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _alembic(database_url: str, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "DATABASE_URL": database_url}
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest_asyncio.fixture()
async def scratch_db():
    """Create an empty scratch database; yield its URL; drop it afterwards."""
    import asyncpg

    base = _base_database_url()
    admin_dsn = _dsn(base).rsplit("/", 1)[0] + "/postgres"

    try:
        admin = await asyncpg.connect(admin_dsn)
    except Exception as exc:
        pytest.skip(f"PostgreSQL not available — migration tests require it: {exc}")

    name = f"axiom_mig_{uuid.uuid4().hex[:10]}"
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()

    scratch_url = base.rsplit("/", 1)[0] + f"/{name}"
    yield scratch_url

    teardown = await asyncpg.connect(admin_dsn)
    try:
        await teardown.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await teardown.close()


class TestFreshDatabaseMigrations:
    @pytest.mark.asyncio
    async def test_upgrade_head_builds_full_schema(self, scratch_db):
        """A truly clean database reaches the full schema from migrations only."""
        import asyncpg

        result = _alembic(scratch_db, "upgrade", "head")
        assert result.returncode == 0, (
            f"alembic upgrade head failed on a clean DB:\n{result.stdout}\n{result.stderr}"
        )

        conn = await asyncpg.connect(_dsn(scratch_db))
        try:
            tables = await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            table_names = {r["table_name"] for r in tables}
            missing = EXPECTED_TABLES - table_names
            assert not missing, f"tables missing after upgrade head: {missing}"

            # pgvector extension present for the embedding column.
            ext = await conn.fetchval(
                "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
            )
            assert ext == 1, "pgvector extension missing after upgrade head"

            # ANN index exists (parity with the runtime DDL).
            idx = await conn.fetchval(
                "SELECT COUNT(*) FROM pg_indexes "
                "WHERE indexname = 'chunk_embeddings_vec_idx'"
            )
            assert idx == 1, "ivfflat index missing after upgrade head"
        finally:
            await conn.close()

        # Downgrade round-trip leaves the database empty again.
        result = _alembic(scratch_db, "downgrade", "base")
        assert result.returncode == 0, result.stderr

        conn = await asyncpg.connect(_dsn(scratch_db))
        try:
            tables = await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            remaining = {r["table_name"] for r in tables}
            # alembic never drops its own version table; every APP table must go.
            assert remaining - {"alembic_version"} == set(), (
                f"app tables survived downgrade: {sorted(remaining)}"
            )
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_upgrade_is_idempotent(self, scratch_db):
        """`upgrade head` twice succeeds the second time."""
        assert _alembic(scratch_db, "upgrade", "head").returncode == 0
        result = _alembic(scratch_db, "upgrade", "head")
        assert result.returncode == 0, result.stderr

    @pytest.mark.asyncio
    async def test_legacy_schema_upgrades_in_place(self, scratch_db):
        """A DB built by the old ad-hoc startup DDL (tables exist, no
        alembic_version) takes `upgrade head` without error."""
        import asyncpg

        conn = await asyncpg.connect(_dsn(scratch_db))
        try:
            # The subset of startup DDL the pre-alembic app ran.
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                "CREATE TABLE chunk_embeddings ("
                "id SERIAL PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE,"
                "source TEXT NOT NULL, content TEXT NOT NULL,"
                "chunk_index INTEGER NOT NULL, embedding vector(1536) NOT NULL,"
                "token_count INTEGER, bm25_score FLOAT,"
                "ingested_at TIMESTAMPTZ DEFAULT NOW())"
            )
            await conn.execute(
                "CREATE TABLE pipeline_traces ("
                "session_id TEXT PRIMARY KEY, trace_data JSONB,"
                "created_at TIMESTAMPTZ DEFAULT NOW())"
            )
        finally:
            await conn.close()

        result = _alembic(scratch_db, "upgrade", "head")
        assert result.returncode == 0, (
            f"legacy-DB upgrade failed:\n{result.stdout}\n{result.stderr}"
        )
