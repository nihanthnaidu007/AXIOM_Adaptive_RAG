import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("POSTGRES_URL", "postgresql+psycopg://test:test@localhost:5432/test")
# Protected endpoints fail closed (503) when API_KEY is unset, so tests run
# with a key configured. Keep in sync with API_KEY in the test modules.
os.environ.setdefault("API_KEY", "test-api-key")

import asyncio

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """The slowapi limiter is process-global and keyed by client address;
    without a reset the 5/minute ingest budget would leak across tests.

    Tolerates environments without the server deps installed (the
    migrations-only CI job imports none of the app)."""
    try:
        import server as server_module
    except ImportError:
        yield
        return
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
    try:
        from axiom.cache.semantic_cache import semantic_cache
        from axiom.retrieval.vector_store import vector_store
    except ImportError:
        return
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
