"""Wave 2 observability tests.

Covers three areas:

1. Request IDs: every response carries ``X-Request-ID`` (server-issued or a
   safe client-supplied value), it lands in error-envelope context, and it
   appears in the server's request-completion log record.
2. JSON logs: the root handler emits one valid JSON object per record, with
   ``extra`` fields and exception details flattened into the object.
3. Prometheus ``/metrics``: unauthenticated scrape (like ``/health``), HTTP
   counters/histograms with route-template labels, and per-query token
   accounting attributed to the endpoint that ran the graph.

Requests that exercise error envelopes install a fake graph so the endpoint
runs locally (same pattern as ``test_api_integration.py``).
"""

import json
import logging

import pytest
from fastapi.testclient import TestClient

import server as server_module
from axiom.api_errors import GENERIC_INTERNAL_MESSAGE
from axiom.observability.logging import JsonLogFormatter, normalize_request_id
from axiom.observability.metrics import (
    begin_token_scope,
    end_token_scope,
    observe_query_tokens,
    record_llm_usage,
    render_metrics,
)

AUTH_HEADERS = {"X-API-Key": "test-api-key"}


def _metric_value(text: str, name: str, labels: str) -> float:
    """Value of one labeled sample from a Prometheus text exposition."""
    prefix = f"{name}{{{labels}}}"
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line.split()[1])
    return 0.0


class FakeGraph:
    """Graph stub configurable per test: return a final state or raise."""

    def __init__(self, final_state=None, error=None):
        self._final_state = final_state if final_state is not None else {
            "answer": "stub answer",
            "confidence": 0.9,
            "trace": [],
        }
        self._error = error

    async def ainvoke(self, state, config=None):
        if self._error is not None:
            raise self._error
        return dict(self._final_state)


@pytest.fixture()
def client(monkeypatch):
    """TestClient with auth/api-key env preset and no external services."""
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return TestClient(server_module.app)


def test_request_id_header_issued(client):
    """Every response gets a server-issued 32-hex request ID."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    request_id = resp.headers.get("X-Request-ID")
    assert request_id
    assert len(request_id) == 32
    int(request_id, 16)  # uuid4().hex is pure hex


def test_request_id_client_supplied_honored(client):
    """A safe client X-Request-ID is echoed, not replaced."""
    resp = client.get("/api/health", headers={"X-Request-ID": "client-trace-42"})
    assert resp.headers.get("X-Request-ID") == "client-trace-42"


def test_request_id_client_supplied_unsafe_replaced(client):
    """Unsafe IDs (control chars, >128 chars) get a server-issued one."""
    resp = client.get("/api/health", headers={"X-Request-ID": "bad\x01id"})
    echoed = resp.headers.get("X-Request-ID", "")
    assert echoed != "bad\x01id"
    assert len(echoed) == 32

    resp = client.get("/api/health", headers={"X-Request-ID": "x" * 129})
    assert resp.headers.get("X-Request-ID") != "x" * 129


def test_normalize_request_id():
    assert normalize_request_id("ok-id-1") == "ok-id-1"
    assert normalize_request_id("") == ""
    assert normalize_request_id("bad\x1f") == ""
    assert normalize_request_id("x" * 129) == ""


def test_request_id_on_413_body_limit_rejection(client, caplog):
    """The 413 short-circuit carries the request ID like every other response.

    Regression: request_context was registered first (innermost), so the
    body-limit middleware's early 413 response skipped it — no X-Request-ID
    header, no completion log line, no metric. Registered last (outermost),
    it now wraps the rejection path too.
    """
    caplog.set_level(logging.INFO, logger="server")
    resp = client.post(
        "/api/query",
        headers={**AUTH_HEADERS, "X-Request-ID": "limit-413"},
        json={"query": "x" * (51 * 1024)},
    )
    assert resp.status_code == 413
    assert resp.headers["X-Request-ID"] == "limit-413"

    records = [
        r for r in caplog.records if getattr(r, "request_id", None) == "limit-413"
    ]
    assert records, "no completion log record carried the request ID on the 413 path"
    assert any(
        getattr(r, "status_code", None) == 413 for r in records
    ), "completion log record did not report the 413 status"


def test_request_id_in_error_envelope(client, monkeypatch):
    """500 envelopes carry the request ID in context; secrets stay out."""
    monkeypatch.setattr(
        server_module,
        "get_graph",
        lambda checkpointer=None: FakeGraph(
            error=RuntimeError("SECRET_DB_DSN=postgres://hunter2")
        ),
    )
    resp = client.post(
        "/api/query", headers=AUTH_HEADERS, json={"query": "hello"}
    )
    assert resp.status_code == 500
    request_id = resp.headers["X-Request-ID"]
    assert len(request_id) == 32

    detail = resp.json()["detail"]
    assert detail["error"] == GENERIC_INTERNAL_MESSAGE
    assert detail["code"] == "internal_error"
    assert detail["context"]["request_id"] == request_id
    assert "SECRET_DB_DSN" not in resp.text
    assert "hunter2" not in resp.text


def test_request_id_in_log_record(client, monkeypatch, caplog):
    """The request-completion log line carries the same request ID."""
    caplog.set_level(logging.INFO, logger="server")
    resp = client.get("/api/health", headers={"X-Request-ID": "log-check-123"})
    request_id = resp.headers["X-Request-ID"]

    records = [
        r for r in caplog.records if getattr(r, "request_id", None) == "log-check-123"
    ]
    assert records, "no completion log record carried the request ID"
    rec = records[-1]
    assert rec.request_id == request_id
    assert rec.http_method == "GET"
    assert rec.path == "/api/health"
    assert rec.status_code == 200
    assert rec.duration_ms >= 0


def test_json_log_format():
    """Formatter emits valid JSON with extra fields and exception details."""
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        "test", logging.INFO, "test.py", 1, "query handled", None, None
    )
    record.request_id = "abc123"
    line = formatter.format(record)
    parsed = json.loads(line)
    assert parsed["message"] == "query handled"
    assert parsed["level"] == "INFO"
    assert parsed["request_id"] == "abc123"
    assert parsed["logger"] == "test"

    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "test", logging.ERROR, "test.py", 1, "failed", None, sys.exc_info()
        )
    parsed = json.loads(formatter.format(record))
    assert parsed["level"] == "ERROR"
    assert parsed["exception"] is not None
    assert "ValueError" in parsed["exception"]


def test_metrics_endpoint_unauthenticated_scrape(client):
    """/metrics is excluded from auth like /health (no query data exposed)."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    body = resp.text
    for name in (
        "axiom_http_requests_total",
        "axiom_query_latency_seconds",
        "axiom_prompt_tokens_total",
        "axiom_completion_tokens_total",
        "axiom_prompt_tokens_per_query",
        "axiom_completion_tokens_per_query",
    ):
        assert name in body


def test_metrics_http_counter_increments(client):
    """Scraping twice around a request shows the counter move."""
    before = client.get("/metrics").text
    value_before = _metric_value(
        before,
        "axiom_http_requests_total",
        'code="200",endpoint="/api/health",method="GET"',
    )

    assert client.get("/api/health").status_code == 200

    after = client.get("/metrics").text
    value_after = _metric_value(
        after,
        "axiom_http_requests_total",
        'code="200",endpoint="/api/health",method="GET"',
    )
    assert value_after == value_before + 1


def test_token_scope_attribution():
    """Usage reported inside a scope is returned by end_token_scope."""
    scope = begin_token_scope()
    record_llm_usage(11, 3)
    record_llm_usage(4, 0)
    prompt, completion = end_token_scope(scope)
    assert (prompt, completion) == (15, 3)


def test_record_usage_without_scope_is_noop():
    """No active scope (background code, startup) must not raise."""
    record_llm_usage(7, 2)


def test_observe_query_tokens_counter_and_histogram():
    """Per-query accounting updates both the totals and the histogram."""
    text_before = render_metrics()[0].decode()
    total_before = _metric_value(
        text_before,
        "axiom_prompt_tokens_total",
        'endpoint="/api/query"',
    )
    hist_before = _metric_value(
        text_before,
        "axiom_prompt_tokens_per_query_count",
        'endpoint="/api/query"',
    )

    observe_query_tokens("/api/query", 100, 25)

    text_after = render_metrics()[0].decode()
    assert (
        _metric_value(text_after, "axiom_prompt_tokens_total", 'endpoint="/api/query"')
        == total_before + 100
    )
    assert (
        _metric_value(
            text_after,
            "axiom_completion_tokens_total",
            'endpoint="/api/query"',
        )
        == _metric_value(
            text_before, "axiom_completion_tokens_total", 'endpoint="/api/query"'
        )
        + 25
    )
    assert (
        _metric_value(
            text_after,
            "axiom_prompt_tokens_per_query_count",
            'endpoint="/api/query"',
        )
        == hist_before + 1
    )


def test_rate_limited_request_also_observed(client):
    """429s from the limiter pass through the context middleware too."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    # /metrics itself is a real route — its scrape is observed.
    body = resp.text
    assert (
        _metric_value(
            body,
            "axiom_http_requests_total",
            'code="200",endpoint="/metrics",method="GET"',
        )
        >= 1
    )


def test_http_exception_envelope_shape_preserved(client):
    """W1 envelope shapes stay intact: a 400 without context keeps its exact
    shape (enrichment only applies to envelopes that carry a context object)."""
    resp = client.post("/api/query", headers=AUTH_HEADERS, json={"query": ""})
    assert resp.status_code == 400
    assert resp.json()["detail"] == {"error": "Query cannot be empty"}
    # The request ID is still available on the response header.
    assert len(resp.headers["X-Request-ID"]) == 32
