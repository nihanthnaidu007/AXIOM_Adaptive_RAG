"""Regression tests for the phase-1 sanitization finding (F2).

Raw exception text must never reach clients — failure surfaces use the
stable sanitized envelopes from ``axiom.api_errors``. Each test injects an
error whose message carries secret-looking tokens and asserts the tokens
never appear in anything a client receives:

- eval job failures: ``_run_eval_background`` stored ``str(exc)`` in the job
  status that ``GET /api/eval/status/{job_id}`` serves verbatim.
- check_cache embedding failures: the raw exception was embedded in trace
  steps served via ``/trace/{session_id}`` and query responses.
"""

import json

import pytest

import server as server_module
from axiom.api_errors import GENERIC_INTERNAL_MESSAGE
from axiom.eval_suite.runner import EvalRunner

AUTH = {"X-API-Key": "test-api-key"}  # keep in sync with tests/conftest.py

SECRET_TOKEN = "sk-super-secret-token-9f3a2b"
SECRET_DSN = "postgres://admin:hunter2@db.internal:5432/axiom"


@pytest.mark.asyncio
async def test_eval_job_failure_error_is_sanitized(client, monkeypatch):
    """A failed eval job reports the stable sanitized message, not str(exc)."""

    async def _boom(self):
        raise RuntimeError(f"auth failed for {SECRET_TOKEN} at {SECRET_DSN}")

    # Keep the endpoint's own background task a no-op; run the real failure
    # path below so the assertion is deterministic, not timing-dependent.
    async def _noop(job_id):
        pass

    monkeypatch.setattr(server_module, "_run_eval_with_semaphore", _noop)
    monkeypatch.setattr(EvalRunner, "_ensure_services", _boom)

    started = await client.post("/api/eval/run", headers=AUTH)
    assert started.status_code == 200, started.text
    job_id = started.json()["job_id"]

    await server_module._run_eval_background(job_id)

    status = await client.get(f"/api/eval/status/{job_id}", headers=AUTH)
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "failed"
    assert body["error"] == GENERIC_INTERNAL_MESSAGE
    assert SECRET_TOKEN not in status.text
    assert "hunter2" not in status.text


@pytest.mark.asyncio
async def test_cache_embedding_error_trace_is_sanitized(monkeypatch):
    """Embedding failures in check_cache must not leak exception text into
    trace steps (served via /trace and query responses)."""
    from axiom.cache.semantic_cache import semantic_cache  # singleton instance
    from axiom.graph.nodes import check_cache as check_cache_module

    async def _boom_embed(text):
        raise RuntimeError(f"embed failed for {SECRET_TOKEN} at {SECRET_DSN}")

    async def _connected():
        return True

    monkeypatch.setattr(check_cache_module, "embed_text", _boom_embed)
    monkeypatch.setattr(semantic_cache, "is_connected", _connected)

    state = {"user_query": "hello"}
    result = await check_cache_module.check_cache_node(state)

    step = result["trace_steps"][-1].model_dump()
    assert "embedding failed" in step["summary"]
    blob = json.dumps(step)
    assert SECRET_TOKEN not in blob
    assert "hunter2" not in blob
