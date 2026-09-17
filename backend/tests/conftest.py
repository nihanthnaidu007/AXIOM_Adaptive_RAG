import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("POSTGRES_URL", "postgresql+psycopg://test:test@localhost:5432/test")
# Protected endpoints fail closed (503) when API_KEY is unset, so tests run
# with a key configured. Keep in sync with API_KEY in test_auth_and_lifecycle.py.
os.environ.setdefault("API_KEY", "test-api-key")

import asyncio

import pytest


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
