"""Chunk-level SSE streaming integration tests (W1).

Complements the node-level stream tests in test_api_integration.py. Covers the
content stream: ordered deltas with request-ID context, the sources/citations
event, semantic-cache-hit streaming, error mid-stream, client-disconnect
teardown, and the fail-closed auth gate on the streaming endpoint.
"""

import asyncio

import pytest
from starlette.requests import Request as StarletteRequest

import server as server_module
from axiom.graph.nodes.check_cache import check_cache_node
from axiom.graph.streaming import get_content_sink
from tests.test_api_integration import AUTH, parse_sse

GENERATE_NODE = "generate_answer"


class FakeStreamingGraph:
    """Compiled-graph stand-in that streams deltas through the ContentSink.

    Mirrors the production shape: an ``on_chain_start`` event for the
    generate_answer node (drives the "generating" status), then deltas
    published into the sink as the LLM would emit them, then the
    ``on_chain_end`` carrying the final state.
    """

    def __init__(self, final_state: dict, deltas: list[str], hold_s: float = 0.0):
        self.final_state = final_state
        self.deltas = deltas
        self.hold_s = hold_s
        self.cancelled = False
        self.completed = False
        self.nodes = {name: None for name in ("check_cache", GENERATE_NODE)}

    async def astream_events(self, state, config, version="v2"):
        sink = get_content_sink()
        assert sink is not None, "streaming graph must run with an installed sink"
        yield {
            "event": "on_chain_start",
            "metadata": {"langgraph_node": GENERATE_NODE},
            "data": {"input": {}},
        }
        for delta in self.deltas:
            sink.publish(delta)
            await asyncio.sleep(0.01)
        if self.hold_s:
            try:
                await asyncio.sleep(self.hold_s)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        self.completed = True
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": GENERATE_NODE},
            "data": {"output": dict(self.final_state)},
        }

    async def astream(self, state, config=None):
        yield {GENERATE_NODE: dict(self.final_state)}

    async def aget_state(self, config):
        return type("Snap", (), {"values": dict(self.final_state)})()


def _final_state() -> dict:
    return {
        "final_answer": "Streamed answer text",
        "confidence": {"label": "PROBABLE", "score": 0.8},
        "classification": {"query_type": "factual", "retrieval_strategy": "hybrid"},
        "retrieval_strategy": "hybrid",
        "ragas_scores": None,
        "scores_history": [],
        "reranked_chunks": [
            {"chunk_id": "c1", "content": "chunk one body", "source": "handbook.pdf", "rerank_score": 0.91},
            {"chunk_id": "c2", "content": "chunk two body", "source": "handbook.pdf", "rerank_score": 0.72},
        ],
        "correction_attempts": 0,
        "correction_history": [],
        "trace_steps": [{"node_name": GENERATE_NODE, "status": "complete", "summary": "ok"}],
        "served_from_cache": False,
        "is_complete": True,
        "error": None,
        "web_search_used": False,
        "web_search_chunks": [],
        "session_id": "unused",
    }


@pytest.fixture()
def install_graph(monkeypatch):
    def _install(graph) -> None:
        monkeypatch.setattr(server_module, "get_graph", lambda checkpointer=None: graph)
        monkeypatch.setattr(
            server_module, "get_graph_node_names", lambda: list(graph.nodes.keys())
        )
        if not hasattr(server_module.app.state, "checkpointer"):
            server_module.app.state.checkpointer = object()

    return _install


@pytest.fixture(autouse=True)
def _clean_stores():
    yield
    server_module._trace_store.clear()


class TestContentStreaming:
    @pytest.mark.asyncio
    async def test_content_events_ordered_with_request_id(self, client, install_graph):
        graph = FakeStreamingGraph(_final_state(), ["Hello ", "streamed ", "world"])
        install_graph(graph)

        r = await client.post("/api/query/stream", json={"query": "stream it"}, headers=AUTH)
        assert r.status_code == 200, r.text
        request_id = r.headers["X-Request-ID"]

        events = parse_sse(r.text)
        types = [e["type"] for e in events]

        # Stage signals announce before their content: retrieving at open,
        # generating when generate_answer starts, then ordered deltas.
        assert types[0] == "status" and events[0]["stage"] == "retrieving"
        assert events[1]["type"] == "status" and events[1]["stage"] == "generating"
        content = [e for e in events if e["type"] == "content"]
        assert [c["delta"] for c in content] == ["Hello ", "streamed ", "world"]
        # Citations arrive after all content, before the terminal event.
        sources = [e for e in events if e["type"] == "sources"]
        assert len(sources) == 1 and sources[0]["sources"][0]["chunk_id"] == "c1"
        assert types[-2:] == ["done", "[DONE]"]

        # Every SSE event carries the request-ID context.
        for e in events:
            if e["type"] != "[DONE]":
                assert e.get("request_id") == request_id, e

        done = next(e for e in events if e["type"] == "done")
        assert done["result"]["final_answer"] == "Streamed answer text"

    @pytest.mark.asyncio
    async def test_error_mid_stream_surfaces_sanitized_error(self, client, install_graph):
        class ExplodingGraph(FakeStreamingGraph):
            async def astream_events(self, state, config, version="v2"):
                sink = get_content_sink()
                assert sink is not None
                sink.publish("partial answer text")
                raise RuntimeError("secret-internal-detail-do-not-leak")
                yield {}  # pragma: no cover — makes this an async generator

            async def astream(self, state, config=None):
                # The endpoint falls back to astream when astream_events fails
                # mid-stream; fail there too so the error surfaces in-band.
                raise RuntimeError("secret-internal-detail-do-not-leak")
                yield {}  # pragma: no cover

        install_graph(ExplodingGraph(_final_state(), []))

        r = await client.post("/api/query/stream", json={"query": "boom"}, headers=AUTH)
        assert r.status_code == 200  # the stream opens; failure arrives in-band

        events = parse_sse(r.text)
        err = next(e for e in events if e.get("type") == "error")
        assert err["code"] == "internal_error"
        assert err.get("request_id") == r.headers["X-Request-ID"]
        assert "secret-internal-detail-do-not-leak" not in r.text
        assert events[-1]["type"] == "[DONE]"
        assert not any(e["type"] == "done" for e in events)


class TestCacheHitStreaming:
    @pytest.mark.asyncio
    async def test_cache_hit_streams_cached_answer_as_content(self, client, install_graph, monkeypatch):
        from axiom.cache.semantic_cache import semantic_cache

        cached = {
            "final_answer": (
                "Cached answer, served warm, with plenty of text so the cache "
                "publisher splits it into multiple streaming deltas for the UI."
            ),
            "similarity": 0.93,
            "retrieval_strategy": "hybrid",
            "correction_attempts": 0,
            "faithfulness_score": 0.88,
            "answer_relevancy": 0.9,
            "context_groundedness": 0.87,
            "composite_score": 0.88,
            "scorer_model": "cached",
            "cache_key": "abc",
        }

        async def fake_search(*args, **kwargs):
            return dict(cached)

        async def fake_is_connected():
            return True

        async def fake_embed(text):
            return [0.1, 0.2]

        monkeypatch.setattr(semantic_cache, "search", fake_search)
        monkeypatch.setattr(semantic_cache, "is_connected", fake_is_connected)
        monkeypatch.setattr("axiom.graph.nodes.check_cache.embed_text", fake_embed)

        class CacheHitGraph:
            """Runs the REAL check_cache node inside the streaming context."""

            def __init__(self):
                self.nodes = {"check_cache": None, GENERATE_NODE: None}

            async def astream_events(self, state, config, version="v2"):
                sink = get_content_sink()
                assert sink is not None
                state = await check_cache_node(state)
                yield {
                    "event": "on_chain_end",
                    "metadata": {"langgraph_node": "check_cache"},
                    "data": {"output": dict(state)},
                }

            async def aget_state(self, config):
                return type("Snap", (), {"values": {}})()

        install_graph(CacheHitGraph())

        r = await client.post("/api/query/stream", json={"query": "warm query"}, headers=AUTH)
        assert r.status_code == 200, r.text

        events = parse_sse(r.text)
        content = [e for e in events if e["type"] == "content"]
        assert content, "cache hit must stream through content events"
        assert "".join(c["delta"] for c in content) == cached["final_answer"]
        assert len(content) > 1, "cached answer should arrive as multiple deltas"

        done = next(e for e in events if e["type"] == "done")
        assert done["result"]["served_from_cache"] is True
        assert done["result"]["ragas_scores"]["evaluation_mode"] == "cached"
        assert events[-1]["type"] == "[DONE]"


class TestDisconnectTeardown:
    @pytest.mark.asyncio
    async def test_client_disconnect_cancels_graph_work(self, client, install_graph, monkeypatch):
        """A client gone mid-stream must not leave the generation running.

        httpx's ASGITransport cannot simulate a real socket disconnect, so the
        disconnect probe itself is stubbed: after the first idle tick the
        endpoint's ``request.is_disconnected()`` reports True — exactly the
        branch a real disconnect takes.
        """
        graph = FakeStreamingGraph(_final_state(), ["partial "], hold_s=30.0)
        install_graph(graph)

        calls = {"n": 0}

        async def fake_is_disconnected(self):
            calls["n"] += 1
            return calls["n"] >= 2  # first tick connected, then gone

        monkeypatch.setattr(StarletteRequest, "is_disconnected", fake_is_disconnected)

        r = await client.post("/api/query/stream", json={"query": "slow gen"}, headers=AUTH)
        assert r.status_code == 200

        assert graph.cancelled is True, "graph work must be cancelled on disconnect"
        assert graph.completed is False, "generation must not run to completion"
        assert "[DONE]" not in r.text, "no terminal frame after a disconnect"
        assert "partial " in r.text  # deltas already emitted did reach the wire

    @pytest.mark.asyncio
    async def test_happy_path_completes_without_disconnect(self, client, install_graph, monkeypatch):
        """Guard for the teardown test: without a disconnect, generation finishes."""
        graph = FakeStreamingGraph(_final_state(), ["quick "])
        install_graph(graph)

        async def never(self):
            return False

        monkeypatch.setattr(StarletteRequest, "is_disconnected", never)

        r = await client.post("/api/query/stream", json={"query": "fast"}, headers=AUTH)
        assert r.status_code == 200
        assert graph.completed is True
        assert graph.cancelled is False
        assert r.text.endswith("data: [DONE]\n\n")


class TestStreamAuthFailClosed:
    @pytest.mark.asyncio
    async def test_stream_without_key_is_401(self, client):
        r = await client.post("/api/query/stream", json={"query": "x"})
        assert r.status_code == 401
        assert r.headers["content-type"].startswith("application/json")

    @pytest.mark.asyncio
    async def test_stream_with_wrong_key_is_401(self, client):
        r = await client.post(
            "/api/query/stream", json={"query": "x"}, headers={"X-API-Key": "wrong"}
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_stream_without_configured_key_fails_closed_503(self, client, monkeypatch):
        cfg = server_module.get_config()
        monkeypatch.setattr(cfg, "api_key", None)

        r = await client.post("/api/query/stream", json={"query": "x"}, headers=AUTH)
        assert r.status_code == 503
        assert "not configured" in r.json()["detail"]["error"]
        assert r.headers["content-type"].startswith("application/json"), (
            "no stream may open when authentication cannot succeed"
        )


class TestSourcesEvent:
    @pytest.mark.asyncio
    async def test_sources_include_web_results_and_dedupe(self, client, install_graph):
        state = _final_state()
        state["web_search_used"] = True
        state["web_search_chunks"] = [
            {"url": "https://example.com/a", "title": "Example A", "score": 0.44, "content": "web body"},
            {"url": "https://example.com/a", "title": "Example A dup", "score": 0.11, "content": "dup"},
        ]
        install_graph(FakeStreamingGraph(state, ["delta "], ))

        r = await client.post("/api/query/stream", json={"query": "citations"}, headers=AUTH)
        assert r.status_code == 200

        sources = next(e for e in parse_sse(r.text) if e["type"] == "sources")
        kinds = [s["kind"] for s in sources["sources"]]
        assert kinds == ["document", "document", "web"], kinds  # dup URL dropped
        web = sources["sources"][-1]
        assert web["url"] == "https://example.com/a"
        assert web["title"] == "Example A"
        assert sources["web_search_used"] is True
