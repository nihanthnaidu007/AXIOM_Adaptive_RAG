"""Integration tests for the full HTTP surface (Wave 1).

Uses the W0 TestClient pattern (httpx ASGITransport, no lifespan) with a fake
LangGraph patched in at ``server.get_graph`` — the pipeline is stubbed, so no
LLM/embedding/network calls happen here. Covers:

- /query: happy path shape, 400 validation, 504 timeout, 500 error envelope
- /query/stream: SSE frame order (node_complete → done → [DONE]), the
  astream_events → astream fallback, error-event envelope with no raw
  exception text
- /trace: populated by a query, 404 shape for unknown sessions
- /session/{id}/state: checkpointed state via the graph, 404 unknown
- /eval/run, /eval/status, /eval/results, /eval/run/stream (stubbed runner)
- /health, request-size middleware, ingest validation (400/413/415)

Error-envelope contract: unexpected failures return
``{"error": "<safe message>", "code": "<machine code>", "context": {...}}``
inside FastAPI's ``detail``; SSE errors emit
``{"type": "error", "code": ..., "message": "<safe message>"}``. Raw exception
text must never reach the client.
"""

import json
import uuid
from types import SimpleNamespace

import pytest

from axiom.api_errors import GENERIC_INTERNAL_MESSAGE

API_KEY = "test-api-key"  # keep in sync with tests/conftest.py
AUTH = {"X-API-Key": API_KEY}

SECRET_DETAIL = "secret-internal-detail-do-not-leak"

NODE_NAMES = ["classify_query", "generate"]


class FakeRagasScores:
    """Pydantic-like stand-in for axiom.graph.state.RAGASScores — the /query
    route reads ``ragas.evaluation_mode`` as an attribute and serializes via
    ``model_dump()``."""

    evaluation_mode = "test"
    faithfulness = 0.9

    def model_dump(self):
        return {"faithfulness": self.faithfulness, "evaluation_mode": self.evaluation_mode}


def _fake_final_state(session_id: str | None = None) -> dict:
    return {
        "final_answer": "stubbed answer",
        "confidence": {"band": "high", "score": 0.9},
        "classification": {"query_type": "factual", "retrieval_strategy": "hybrid"},
        "retrieval_strategy": "hybrid",
        "ragas_scores": FakeRagasScores(),
        "scores_history": [],
        "reranked_chunks": [{"chunk_id": "c1", "content": "stub"}],
        "correction_attempts": 0,
        "correction_history": [],
        "trace_steps": [
            {"node_name": "classify_query", "status": "ok", "summary": "classified"},
            {"node_name": "generate", "status": "ok", "summary": "generated"},
        ],
        "served_from_cache": False,
        "is_complete": True,
        "error": None,
        **({"session_id": session_id} if session_id else {}),
    }


class FakeGraph:
    """Stands in for the compiled LangGraph.

    Args:
        final_state: returned by ainvoke / aget_state.
        invoke_error: exception raised by ainvoke (drives /query 500).
        stream_error: exception raised by BOTH astream_events and astream
            (drives the SSE error event).
        events_error: exception raised by astream_events only (drives the
            fallback to astream).
    """

    def __init__(self, final_state: dict, invoke_error=None, stream_error=None, events_error=None):
        self.final_state = final_state
        self.nodes = {name: None for name in NODE_NAMES}
        self.invoke_error = invoke_error
        self.stream_error = stream_error
        self.events_error = events_error

    async def ainvoke(self, state, config=None):
        if self.invoke_error is not None:
            raise self.invoke_error
        return dict(self.final_state)

    async def astream_events(self, state, config, version="v2"):
        if self.events_error is not None:
            raise self.events_error
        if self.stream_error is not None:
            raise self.stream_error
        steps = self.final_state["trace_steps"]
        for i, step in enumerate(steps):
            output = dict(self.final_state)
            output["trace_steps"] = steps[: i + 1]
            yield {
                "event": "on_chain_end",
                "metadata": {"langgraph_node": step["node_name"]},
                "data": {"output": output},
            }

    async def astream(self, state, config=None):
        if self.stream_error is not None:
            raise self.stream_error
        for i, step in enumerate(self.final_state["trace_steps"]):
            output = dict(self.final_state)
            output["trace_steps"] = self.final_state["trace_steps"][: i + 1]
            yield {step["node_name"]: output}

    async def aget_state(self, config):
        return SimpleNamespace(values=dict(self.final_state))


def parse_sse(text: str) -> list[dict]:
    """Parse the server's ``data: <json>\\n\\n`` frames into payloads."""
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data: "), f"malformed SSE frame: {block!r}"
        payload = block[len("data: "):]
        if payload == "[DONE]":
            events.append({"type": "[DONE]"})
        else:
            events.append(json.loads(payload))
    return events


@pytest.fixture()
def install_graph(monkeypatch):
    """Patch the graph factory + node names and stub the checkpointer."""

    def _install(graph: FakeGraph) -> FakeGraph:
        import server as server_module

        monkeypatch.setattr(server_module, "get_graph", lambda checkpointer=None: graph)
        monkeypatch.setattr(
            server_module, "get_graph_node_names", lambda: list(graph.nodes.keys())
        )
        if not hasattr(server_module.app.state, "checkpointer"):
            server_module.app.state.checkpointer = object()
        return graph

    return _install


@pytest.fixture(autouse=True)
def _clean_in_memory_stores():
    """Keep module-level stores from leaking between tests."""
    import server as server_module

    yield
    server_module._trace_store.clear()
    server_module._eval_jobs.clear()


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    @pytest.mark.asyncio
    async def test_query_happy_path_full_shape(self, client, install_graph):
        session = str(uuid.uuid4())
        install_graph(FakeGraph(_fake_final_state(session)))

        r = await client.post(
            "/api/query",
            json={"query": "what is rag?", "session_id": session},
            headers=AUTH,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["session_id"] == session
        assert body["final_answer"] == "stubbed answer"
        assert body["retrieval_strategy"] == "hybrid"
        assert body["is_complete"] is True
        assert body["served_from_cache"] is False
        assert [s["node_name"] for s in body["trace_steps"]] == NODE_NAMES
        assert body["evaluation_mode"] == "test"
        assert "system_health" in body and "total_latency_ms" in body

    @pytest.mark.asyncio
    async def test_query_generates_session_id_when_absent(self, client, install_graph):
        install_graph(FakeGraph(_fake_final_state()))
        r = await client.post("/api/query", json={"query": "hello"}, headers=AUTH)
        assert r.status_code == 200
        # Echoed session id is a UUID the client can use for /trace.
        uuid.UUID(r.json()["session_id"])  # raises if not a UUID

    @pytest.mark.asyncio
    async def test_query_validation_errors(self, client):
        r = await client.post("/api/query", json={"query": "   "}, headers=AUTH)
        assert r.status_code == 400
        assert r.json()["detail"]["error"] == "Query cannot be empty"

        r = await client.post(
            "/api/query", json={"query": "x" * 5000}, headers=AUTH
        )
        assert r.status_code == 400
        assert "maximum" in r.json()["detail"]["error"].lower()

        r = await client.post(
            "/api/query",
            json={"query": "hello", "session_id": "not-a-uuid"},
            headers=AUTH,
        )
        assert r.status_code == 400
        assert "uuid" in r.json()["detail"]["error"].lower()

    @pytest.mark.asyncio
    async def test_query_failure_returns_sanitized_envelope(self, client, install_graph):
        install_graph(FakeGraph(_fake_final_state(), invoke_error=RuntimeError(SECRET_DETAIL)))

        r = await client.post("/api/query", json={"query": "boom"}, headers=AUTH)
        assert r.status_code == 500
        detail = r.json()["detail"]
        assert detail["error"] == GENERIC_INTERNAL_MESSAGE
        assert detail["code"] == "internal_error"
        assert "session_id" in detail["context"]
        assert SECRET_DETAIL not in r.text, "raw exception text leaked to client"

    @pytest.mark.asyncio
    async def test_query_timeout_is_504_not_500(self, client, install_graph, monkeypatch):
        import asyncio

        import server as server_module

        class SlowGraph(FakeGraph):
            async def ainvoke(self, state, config=None):
                await asyncio.sleep(0.5)
                return dict(self.final_state)

        install_graph(SlowGraph(_fake_final_state()))
        monkeypatch.setattr(server_module, "QUERY_GRAPH_TIMEOUT_SEC", 0.05)

        r = await client.post("/api/query", json={"query": "slow"}, headers=AUTH)
        assert r.status_code == 504, (
            "the intentional 504 timeout envelope must not be re-wrapped as a 500"
        )
        assert "timed out" in r.json()["detail"]["error"].lower()

    @pytest.mark.asyncio
    async def test_trace_populated_after_query(self, client, install_graph):
        session = str(uuid.uuid4())
        install_graph(FakeGraph(_fake_final_state(session)))

        r = await client.post(
            "/api/query", json={"query": "hello", "session_id": session}, headers=AUTH
        )
        assert r.status_code == 200

        r = await client.get(f"/api/trace/{session}", headers=AUTH)
        assert r.status_code == 200
        assert [s["node_name"] for s in r.json()["trace_steps"]] == NODE_NAMES


# ---------------------------------------------------------------------------
# /query/stream (SSE)
# ---------------------------------------------------------------------------


class TestQueryStreamEndpoint:
    @pytest.mark.asyncio
    async def test_stream_emits_frames_in_order(self, client, install_graph):
        install_graph(FakeGraph(_fake_final_state()))

        r = await client.post("/api/query/stream", json={"query": "stream me"}, headers=AUTH)
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/event-stream")

        events = parse_sse(r.text)
        types = [e["type"] for e in events]
        # status → per-node trace → sources (citations) → done → [DONE]
        assert types == [
            "status", "node_complete", "node_complete", "sources", "done", "[DONE]"
        ], types

        status = events[0]
        assert status["stage"] == "retrieving"
        assert status["request_id"] == r.headers["X-Request-ID"]

        done = events[-2]
        assert done["result"]["final_answer"] == "stubbed answer"
        assert done["result"]["is_complete"] is True

        sources = events[-3]
        assert sources["sources"][0]["kind"] == "document"
        assert sources["sources"][0]["chunk_id"] == "c1"
        assert sources["request_id"] == r.headers["X-Request-ID"]

    @pytest.mark.asyncio
    async def test_stream_falls_back_to_astream(self, client, install_graph):
        install_graph(
            FakeGraph(
                _fake_final_state(),
                events_error=RuntimeError("astream_events unavailable"),
            )
        )

        r = await client.post("/api/query/stream", json={"query": "fallback"}, headers=AUTH)
        assert r.status_code == 200
        events = parse_sse(r.text)
        types = [e["type"] for e in events]
        assert types[0] == "status"
        assert "node_complete" in types
        assert "done" in types and types[-1] == "[DONE]"
        assert SECRET_DETAIL not in r.text

    @pytest.mark.asyncio
    async def test_stream_failure_is_sanitized_error_event(self, client, install_graph):
        install_graph(
            FakeGraph(_fake_final_state(), stream_error=RuntimeError(SECRET_DETAIL))
        )

        r = await client.post("/api/query/stream", json={"query": "boom"}, headers=AUTH)
        assert r.status_code == 200  # stream starts; failure arrives as an event
        events = parse_sse(r.text)
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 1
        err = error_events[0]
        assert err["code"] == "internal_error"
        assert err["message"] == GENERIC_INTERNAL_MESSAGE
        assert SECRET_DETAIL not in r.text, "raw exception text leaked in SSE stream"
        assert events[-1]["type"] == "[DONE]", "stream must always terminate with [DONE]"

    @pytest.mark.asyncio
    async def test_stream_validation_before_streaming(self, client):
        r = await client.post("/api/query/stream", json={"query": ""}, headers=AUTH)
        assert r.status_code == 400
        assert r.json()["detail"]["error"] == "Query cannot be empty"


# ---------------------------------------------------------------------------
# /trace, /session state
# ---------------------------------------------------------------------------


class TestTraceAndSessionState:
    @pytest.mark.asyncio
    async def test_trace_unknown_session_is_404(self, client):
        r = await client.get(f"/api/trace/{uuid.uuid4()}", headers=AUTH)
        assert r.status_code == 404
        assert "No trace found" in r.json()["detail"]["error"]

    @pytest.mark.asyncio
    async def test_session_state_from_checkpoint(self, client, install_graph):
        session = str(uuid.uuid4())
        install_graph(FakeGraph(_fake_final_state(session)))

        r = await client.get(f"/api/session/{session}/state", headers=AUTH)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["has_state"] is True
        assert body["is_complete"] is True
        assert body["retrieval_strategy"] == "hybrid"

    @pytest.mark.asyncio
    async def test_session_state_unknown_is_404(self, client, install_graph):
        class EmptyGraph(FakeGraph):
            async def aget_state(self, config):
                return SimpleNamespace(values=None)

        install_graph(EmptyGraph(_fake_final_state()))
        r = await client.get(f"/api/session/{uuid.uuid4()}/state", headers=AUTH)
        assert r.status_code == 404
        assert "No state found" in r.json()["detail"]["error"]


# ---------------------------------------------------------------------------
# Eval endpoints
# ---------------------------------------------------------------------------


class FakeEvalRunner:
    results = None

    async def _ensure_services(self):
        pass

    async def run_single(self, bq, session_id):
        return {
            "is_complete": True,
            "actual_strategy": "hybrid",
            "ragas_scores": {"faithfulness": 0.9},
            "error": None,
        }

    def _compute_aggregate(self, total_s, results):
        return {"queries": len(results)}

    def save_results(self, aggregate):
        pass


class TestEvalEndpoints:
    @pytest.mark.asyncio
    async def test_eval_run_starts_background_job(self, client, monkeypatch):
        import server as server_module

        async def _noop(job_id):
            pass

        monkeypatch.setattr(server_module, "_run_eval_with_semaphore", _noop)

        r = await client.post("/api/eval/run", headers=AUTH)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "started"
        assert body["poll_url"] == f"/api/eval/status/{body['job_id']}"

        r = await client.get(f"/api/eval/status/{body['job_id']}", headers=AUTH)
        assert r.status_code == 200
        assert r.json()["status"] == "running"

    @pytest.mark.asyncio
    async def test_eval_status_unknown_job_is_404(self, client):
        r = await client.get("/api/eval/status/nope", headers=AUTH)
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_eval_results_404_shape(self, client):
        r = await client.get("/api/eval/results", headers=AUTH)
        assert r.status_code == 404
        assert "No eval results" in r.json()["detail"]["error"]

    @pytest.mark.asyncio
    async def test_eval_stream_emits_progress_and_final(self, client, monkeypatch):
        monkeypatch.setattr(
            "axiom.eval_suite.runner.EvalRunner", FakeEvalRunner
        )
        monkeypatch.setattr(
            "axiom.eval_suite.benchmark.BENCHMARK_QUERIES",
            [{"query": "q1"}, {"query": "q2"}],
        )

        r = await client.post("/api/eval/run/stream", headers=AUTH)
        assert r.status_code == 200
        events = parse_sse(r.text)
        progress = [e for e in events if "final" not in e]
        assert len(progress) == 2
        assert progress[0]["progress"] == "1/2"
        assert events[-1]["final"] is True
        assert events[-1]["aggregate"] == {"queries": 2}

    @pytest.mark.asyncio
    async def test_eval_stream_failure_is_sanitized(self, client, monkeypatch):
        class ExplodingRunner(FakeEvalRunner):
            async def run_single(self, bq, session_id):
                raise RuntimeError(SECRET_DETAIL)

        monkeypatch.setattr("axiom.eval_suite.runner.EvalRunner", ExplodingRunner)
        monkeypatch.setattr(
            "axiom.eval_suite.benchmark.BENCHMARK_QUERIES", [{"query": "q1"}]
        )

        r = await client.post("/api/eval/run/stream", headers=AUTH)
        assert r.status_code == 200
        events = parse_sse(r.text)
        err = events[-1]
        assert err["type"] == "error"
        assert err["code"] == "internal_error"
        assert err["message"] == GENERIC_INTERNAL_MESSAGE
        assert SECRET_DETAIL not in r.text


# ---------------------------------------------------------------------------
# Middleware, health, ingest validation
# ---------------------------------------------------------------------------


class TestMiddlewareAndValidation:
    @pytest.mark.asyncio
    async def test_health_public_and_shaped(self, client):
        r = await client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        for key in ("status", "stub_mode", "system_health", "index_status"):
            assert key in body

    @pytest.mark.asyncio
    async def test_oversized_json_body_is_413(self, client):
        r = await client.post(
            "/api/query", json={"query": "x" * (51 * 1024)}, headers=AUTH
        )
        assert r.status_code == 413
        assert "too large" in r.json()["error"].lower()

    @pytest.mark.asyncio
    async def test_ingest_rejects_disallowed_extension(self, client):
        r = await client.post(
            "/api/ingest",
            files={"file": ("payload.exe", b"binary", "application/octet-stream")},
            headers=AUTH,
        )
        assert r.status_code == 400
        assert "Unsupported file type" in r.json()["detail"]["error"]

    @pytest.mark.asyncio
    async def test_ingest_rejects_empty_file(self, client):
        r = await client.post(
            "/api/ingest",
            files={"file": ("empty.txt", b"", "text/plain")},
            headers=AUTH,
        )
        assert r.status_code == 400
        assert "empty" in r.json()["detail"]["error"].lower()

    @pytest.mark.asyncio
    async def test_ingest_rejects_non_document_mime(self, client):
        # ZIP bytes under a .txt name pass the extension gate but fail content
        # sniffing — python-magic identifies the real type.
        zip_bytes = b"PK\x03\x04" + b"\x00" * 512
        r = await client.post(
            "/api/ingest",
            files={"file": ("payload.txt", zip_bytes, "text/plain")},
            headers=AUTH,
        )
        assert r.status_code == 415
        assert "Unsupported file type" in r.json()["detail"]["error"]
