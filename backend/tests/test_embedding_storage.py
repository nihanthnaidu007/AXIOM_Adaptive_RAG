"""Wave 3 embedding-storage tests (spec D2.4): migration chain, startup
dimension check, and the re-embed tool's dimension resolution.

No live PostgreSQL: the migration-chain test walks Alembic's revision graph
via ScriptDirectory; the dimension resolution tests import the actual
migration module (filename starts with a digit, so importlib it is); the
vector-store startup check uses a fake async engine that answers the
pg_attribute probe.
"""

import asyncio
import importlib.util
from pathlib import Path

import pytest
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory

import axiom.retrieval.vector_store as vs_module
from axiom.config import AxiomConfig

_MIGRATION_FILE = Path("alembic/versions/9d3b7c4e2f61_embedding_dimensions.py")


def _migration_module():
    """Load the W3 migration by path — alembic/ is not an importable package."""
    spec = importlib.util.spec_from_file_location("w3_dimension_migration", _MIGRATION_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _alembic_script() -> ScriptDirectory:
    return ScriptDirectory.from_config(AlembicConfig("alembic.ini"))


class TestMigrationChain:
    def test_single_head_stacked_after_w2(self):
        """The W3 dimension revision must stack linearly on W2's head —
        a forked chain would break `alembic upgrade head` for every deploy."""
        script = _alembic_script()
        heads = [h.revision for h in script.get_revisions("heads")]
        assert heads == ["9d3b7c4e2f61"], f"Expected one head, got {heads}"
        w3 = script.get_revision("9d3b7c4e2f61")
        assert w3.down_revision == "f2a9c4e7b1d8"

    def test_full_walk_has_no_gaps(self):
        revisions = {rev.revision for rev in _alembic_script().walk_revisions()}
        assert revisions == {
            "22496c2e6b17",
            "e7f3a9c1d2b4",
            "c4d8e2f6a9b1",
            "f2a9c4e7b1d8",
            "9d3b7c4e2f61",
        }


class TestMigrationDimensionResolution:
    """The migration resolves the target width from the process environment
    (frozen copy of the config rules — migrations cannot import app code)."""

    def test_cloud_default(self, monkeypatch):
        migration = _migration_module()
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
        assert migration._resolve_target_dimension() == 1536

    def test_cloud_override(self, monkeypatch):
        migration = _migration_module()
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")
        assert migration._resolve_target_dimension() == 3072

    def test_local_default(self, monkeypatch):
        migration = _migration_module()
        monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
        monkeypatch.delenv("LOCAL_EMBEDDING_DIMENSIONS", raising=False)
        assert migration._resolve_target_dimension() == 768

    def test_local_override_case_insensitive(self, monkeypatch):
        migration = _migration_module()
        monkeypatch.setenv("EMBEDDING_PROVIDER", "LOCAL")
        monkeypatch.setenv("LOCAL_EMBEDDING_DIMENSIONS", "1024")
        assert migration._resolve_target_dimension() == 1024

    def test_invalid_value_raises(self, monkeypatch):
        migration = _migration_module()
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "wide")
        with pytest.raises(RuntimeError, match="not an integer"):
            migration._resolve_target_dimension()

    def test_nonpositive_raises(self, monkeypatch):
        migration = _migration_module()
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "0")
        with pytest.raises(RuntimeError, match="positive"):
            migration._resolve_target_dimension()


class TestVectorStoreStartupDimensionCheck:
    """connect() must fail visibly when the stored column width disagrees
    with the configured embedding identity."""

    def _connect_with(self, monkeypatch, atttypmod, cfg):
        class FakeResult:
            def __init__(self, rows):
                self._rows = rows

            def fetchone(self):
                return self._rows[0] if self._rows else None

        class FakeConn:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

            async def execute(self, stmt):
                if "atttypmod" in str(stmt):
                    return FakeResult([(atttypmod,)])
                return FakeResult([])

        class FakeEngine:
            def begin(self):
                return FakeConn()

            async def dispose(self):
                pass

        monkeypatch.setattr(vs_module, "create_async_engine", lambda *a, **k: FakeEngine())
        monkeypatch.setattr(vs_module, "get_config", lambda: cfg)
        store = vs_module.VectorStore()
        ok = asyncio.run(store.connect())
        return ok, store

    def test_mismatch_fails_closed(self, monkeypatch):
        cfg = AxiomConfig(
            llm_provider="local",
            embedding_provider="local",
            local_embedding_dimensions=768,
            anthropic_api_key="",
            openai_api_key="",
            postgres_url="postgresql://t",
        )
        ok, store = self._connect_with(monkeypatch, atttypmod=1536, cfg=cfg)
        assert ok is False
        assert store._connected is False

    def test_matching_width_connects(self, monkeypatch):
        cfg = AxiomConfig(
            anthropic_api_key="k",
            openai_api_key="sk-x",
            postgres_url="postgresql://t",
        )
        ok, store = self._connect_with(monkeypatch, atttypmod=1536, cfg=cfg)
        assert ok is True
        assert store._connected is True
