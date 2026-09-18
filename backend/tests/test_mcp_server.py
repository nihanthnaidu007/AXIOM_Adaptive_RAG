"""Wave 3 MCP stdio server tests (spec D3.3).

Protocol-level coverage of backend/mcp_server.py — stdlib-only server,
tested through the same dispatch loop a host drives: initialize handshake,
tools/list schema, tools/call success and fail-closed error envelopes, JSON-RPC
error codes, notification silence, and source extraction. The pipeline bridge
is patched at `_run_axiom_query`, so no graph, provider, or daemon is touched.
"""

import json
from unittest.mock import patch

import mcp_server as mcp


async def _collect(responses: list, payload: dict) -> None:
    responses.append(payload)


def _dispatch(line: str):
    """Run one line through the dispatcher; return the emitted response."""
    responses: list = []
    import asyncio

    asyncio.run(mcp._dispatch(line, lambda payload: _collect(responses, payload)))
    assert len(responses) <= 1, "a notification or malformed frame must not double-emit"
    return responses[0] if responses else None


class TestHandshake:
    def test_initialize_returns_protocol_capabilities_and_info(self):
        response = _dispatch(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            )
        )
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        result = response["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == "axiom"
        assert "tools" in result["capabilities"]

    def test_initialize_defaults_when_no_version_requested(self):
        response = _dispatch(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "initialize"}))
        assert response["result"]["protocolVersion"] == mcp.MCP_PROTOCOL_VERSION

    def test_initialized_notification_is_silent(self):
        response = _dispatch(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        assert response is None, "notifications must never be answered"

    def test_ping_returns_empty_result(self):
        response = _dispatch(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}))
        assert response["result"] == {}


class TestToolsList:
    def test_lists_axiom_query_with_schema(self):
        response = _dispatch(json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/list"}))
        tools = response["result"]["tools"]
        assert [t["name"] for t in tools] == ["axiom_query"]
        schema = tools[0]["inputSchema"]
        assert schema["required"] == ["query"]
        assert schema["properties"]["query"]["type"] == "string"


class TestToolsCall:
    def test_success_returns_answer_and_sources(self):
        result_payload = {
            "answer": "Retries are transient-only.",
            "session_id": "00000000-0000-0000-0000-000000000001",
            "sources": [
                {
                    "kind": "document",
                    "source": "spec.md",
                    "chunk_id": "c1",
                    "score": 0.9,
                    "snippet": "Retries are transient-only.",
                }
            ],
        }

        async def fake_query(query, session_id):
            return result_payload

        with patch.object(mcp, "_run_axiom_query", fake_query):
            response = _dispatch(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 5,
                        "method": "tools/call",
                        "params": {
                            "name": "axiom_query",
                            "arguments": {"query": "What about retries?"},
                        },
                    }
                )
            )
        assert response["result"]["isError"] is False
        blocks = response["result"]["content"]
        assert blocks[0]["type"] == "text"
        assert "Retries are transient-only." in blocks[0]["text"]
        sources = json.loads(blocks[1]["text"])["sources"]
        assert sources[0]["source"] == "spec.md"

    def test_pipeline_failure_is_sanitized_fail_closed(self):
        """A raw exception must cross the MCP boundary only as a sanitized
        envelope — no model names, hosts, or exception text."""

        async def boom(query, session_id):
            raise RuntimeError("http://secret-host:11434/api/chat returned secret plan")

        with patch.object(mcp, "_run_axiom_query", boom):
            response = _dispatch(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 6,
                        "method": "tools/call",
                        "params": {"name": "axiom_query", "arguments": {"query": "q"}},
                    }
                )
            )
        result = response["result"]
        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert "secret-host" not in text and "RuntimeError" not in text
        assert text == mcp._MSG_QUERY_FAILED

    def test_unavailable_pipeline_returns_sanitized_message(self):
        async def unavailable(query, session_id):
            raise mcp.PipelineUnavailableError(mcp._MSG_NOT_CONFIGURED)

        with patch.object(mcp, "_run_axiom_query", unavailable):
            response = _dispatch(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 7,
                        "method": "tools/call",
                        "params": {"name": "axiom_query", "arguments": {"query": "q"}},
                    }
                )
            )
        assert response["result"]["isError"] is True
        assert response["result"]["content"][0]["text"] == mcp._MSG_NOT_CONFIGURED

    def test_empty_query_rejected_as_protocol_error(self):
        response = _dispatch(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 8,
                    "method": "tools/call",
                    "params": {"name": "axiom_query", "arguments": {"query": "   "}},
                }
            )
        )
        assert response["error"]["code"] == mcp._JSONRPC_INVALID_REQUEST
        assert response["error"]["message"] == mcp._MSG_EMPTY_QUERY

    def test_oversized_query_rejected(self):
        response = _dispatch(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 9,
                    "method": "tools/call",
                    "params": {"name": "axiom_query", "arguments": {"query": "x" * 2001}},
                }
            )
        )
        assert response["error"]["message"] == mcp._MSG_LONG_QUERY

    def test_bad_session_id_rejected(self):
        response = _dispatch(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 10,
                    "method": "tools/call",
                    "params": {
                        "name": "axiom_query",
                        "arguments": {"query": "q", "session_id": "not-a-uuid"},
                    },
                }
            )
        )
        assert response["error"]["message"] == mcp._MSG_BAD_SESSION


class TestProtocolErrors:
    def test_unknown_method(self):
        response = _dispatch(json.dumps({"jsonrpc": "2.0", "id": 11, "method": "resources/list"}))
        assert response["error"]["code"] == mcp._JSONRPC_METHOD_NOT_FOUND

    def test_malformed_json(self):
        response = _dispatch("{not json")
        assert response["error"]["code"] == mcp._JSONRPC_PARSE_ERROR
        assert response["id"] is None

    def test_missing_method(self):
        response = _dispatch(json.dumps({"jsonrpc": "2.0", "id": 12}))
        assert response["error"]["code"] == mcp._JSONRPC_INVALID_REQUEST


class TestSourceExtraction:
    def test_document_chunks_then_web_results(self):
        from axiom.graph.state import RetrievedChunk

        state = {
            "reranked_chunks": [
                RetrievedChunk(
                    chunk_id="c1",
                    source="spec.md",
                    content="x" * 300,
                    rerank_score=0.9,
                )
            ],
            "web_search_used": True,
            "web_search_chunks": [
                {"url": "https://example.com/a", "content": "web content", "score": 0.4}
            ],
        }
        sources = mcp._sources_from_state(state)
        assert sources[0]["kind"] == "document"
        assert sources[0]["source"] == "spec.md"
        assert len(sources[0]["snippet"]) == 200  # truncated, not full content
        assert sources[1]["kind"] == "web"
        assert sources[1]["source"] == "https://example.com/a"

    def test_web_results_omitted_when_unused(self):
        sources = mcp._sources_from_state({"reranked_chunks": [], "web_search_used": False})
        assert sources == []


class TestTrustBoundary:
    def test_server_has_no_network_listener_surface(self):
        """The module must not start servers or open sockets — its only
        transport is the stdio pair (SECURITY.md documents this)."""
        source = open("mcp_server.py", encoding="utf-8").read()
        assert "uvicorn" not in source
        assert "start_server" not in source
        assert ".bind(" not in source and ".listen(" not in source
        assert "socketserver" not in source
