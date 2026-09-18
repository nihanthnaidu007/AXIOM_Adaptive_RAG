"""AXIOM MCP stdio server — read-only query tool over newline-delimited JSON-RPC.

Wave 3 (D3). Zero new dependencies: the protocol layer uses only the
standard library (json/sys/asyncio); pipeline imports happen lazily inside
the query handler so protocol clients can complete a handshake without
booting LangGraph or any provider client.

Transport contract (Model Context Protocol, stdio):
  - One JSON-RPC 2.0 message per stdin line; one response line per request on
    stdout, flushed immediately. stdout carries NOTHING else — logs and
    diagnostics go to stderr only.
  - Methods: ``initialize``, ``notifications/initialized`` (notification —
    never answered), ``ping``, ``tools/list``, ``tools/call``.
  - Unknown methods → JSON-RPC error -32601; malformed JSON → -32700.

Tool surface: a single read-only tool, ``axiom_query`` — run the AXIOM
retrieval pipeline for one question and return its answer plus grounded
sources. There is deliberately no write surface, no file access, and no
network listener: this server speaks only over the stdin/stdout pair its
host opened (trust boundary documented in SECURITY.md).

Fail-closed semantics (mirroring the HTTP API):
  - Configuration that fails provider validation (missing keys in cloud
    mode, bad provider names) → the tool returns ``isError: true`` with a
    sanitized message instead of a degraded answer.
  - Pipeline failures (retrieval unavailable, generation down, timeout) →
    ``isError: true`` with a sanitized message; the raw exception and its
    detail go to server logs on stderr only, never into the response.
  - Empty/oversized queries → sanitized error, mirroring the HTTP 400s.
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Any, Callable, Dict, List

logger = logging.getLogger("axiom.mcp")

SERVER_NAME = "axiom"
SERVER_VERSION = "1.0.0"
MCP_PROTOCOL_VERSION = "2024-11-05"
QUERY_GRAPH_TIMEOUT_SEC = float(os.environ.get("QUERY_GRAPH_TIMEOUT_SEC", "180"))

_JSONRPC_PARSE_ERROR = -32700
_JSONRPC_METHOD_NOT_FOUND = -32601
_JSONRPC_INVALID_REQUEST = -32600
_JSONRPC_INTERNAL_ERROR = -32603

# Sanitized client-facing messages — raw exception detail stays on stderr.
_MSG_NOT_CONFIGURED = (
    "AXIOM is not configured to serve queries. Check the provider configuration and server logs."
)
_MSG_QUERY_FAILED = "Query failed. Check the server logs for details."
_MSG_QUERY_TIMEOUT = "Query timed out. Try a narrower question or check provider health."
_MSG_EMPTY_QUERY = "Query cannot be empty."
_MSG_LONG_QUERY = "Query too long - maximum 2000 characters."
_MSG_BAD_SESSION = "session_id must be a valid UUID."

AXIOM_QUERY_TOOL = {
    "name": "axiom_query",
    "description": (
        "Ask the AXIOM document corpus a question. Runs the full adaptive "
        "RAG pipeline (hybrid retrieval, reranking, optional web fallback, "
        "evaluation) and returns the grounded answer with its sources. "
        "Read-only."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question to answer against the indexed corpus.",
                "minLength": 1,
                "maxLength": 2000,
            },
            "session_id": {
                "type": "string",
                "description": "Optional conversation id (UUID) to continue a thread.",
            },
        },
        "required": ["query"],
    },
}


class McpRequestError(Exception):
    """A JSON-RPC protocol error carrying its error code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Pipeline bridge — lazy imports keep protocol tests graph-free
# ---------------------------------------------------------------------------


async def _run_axiom_query(query: str, session_id: str) -> Dict[str, Any]:
    """Run one query through the pipeline, mirroring the HTTP /api/query path.

    Returns a dict with ``answer`` and ``sources``; raises on any failure
    (config problems, retrieval/generation errors, timeouts).
    """
    from axiom.graph.graph import get_graph
    from axiom.graph.state import create_initial_state

    cfg_valid = _validate_provider_config()
    if not cfg_valid:
        raise PipelineUnavailableError(_MSG_NOT_CONFIGURED)

    initial_state = create_initial_state(user_query=query, session_id=session_id)
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}
    try:
        final_state = await asyncio.wait_for(
            graph.ainvoke(initial_state, config=config),
            timeout=QUERY_GRAPH_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        raise PipelineUnavailableError(_MSG_QUERY_TIMEOUT) from None

    return {
        "answer": final_state.get("final_answer", ""),
        "session_id": session_id,
        "sources": _sources_from_state(final_state),
    }


class PipelineUnavailableError(RuntimeError):
    """Query cannot be served — surfaced to the client with its (sanitized)
    message and logged in full on the server side."""


def _validate_provider_config() -> bool:
    try:
        from axiom.config import get_config

        # AxiomConfig runs a @model_validator(mode="after") at construction:
        # building it raises when required provider keys are missing.
        get_config()
        return True
    except Exception as exc:
        logger.error("Provider configuration invalid: %s", exc)
        return False


def _sources_from_state(final_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract grounded sources from the final graph state.

    Document chunks come first (they carry retrieval scores), then web
    fallback results when the pipeline used them. Snippets are truncated —
    the MCP client gets provenance, not the whole corpus.
    """
    sources: List[Dict[str, Any]] = []
    for chunk in final_state.get("reranked_chunks", []) or []:
        if hasattr(chunk, "model_dump"):
            chunk = chunk.model_dump()
        score = chunk.get("rerank_score") or chunk.get("vector_score") or chunk.get("bm25_score")
        sources.append(
            {
                "kind": "document",
                "source": chunk.get("source", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "score": float(score) if score is not None else None,
                "snippet": (chunk.get("content", "") or "")[:200],
            }
        )
    if final_state.get("web_search_used"):
        for result in final_state.get("web_search_chunks", []) or []:
            sources.append(
                {
                    "kind": "web",
                    "source": result.get("url", ""),
                    "chunk_id": "",
                    "score": result.get("score"),
                    "snippet": (result.get("content", "") or "")[:200],
                }
            )
    return sources


# ---------------------------------------------------------------------------
# Request handlers — pure dispatch, each returns a JSON-RPC response payload
# ---------------------------------------------------------------------------


def _handle_initialize(params: Dict[str, Any]) -> Dict[str, Any]:
    requested = params.get("protocolVersion")
    return {
        "protocolVersion": requested
        if isinstance(requested, str) and requested
        else MCP_PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
    }


def _handle_tools_list(params: Dict[str, Any]) -> Dict[str, Any]:
    return {"tools": [AXIOM_QUERY_TOOL]}


def _extract_query_args(arguments: Any) -> tuple[str, str]:
    if not isinstance(arguments, dict):
        raise McpRequestError(_JSONRPC_INVALID_REQUEST, "params.arguments must be an object")
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise McpRequestError(_JSONRPC_INVALID_REQUEST, _MSG_EMPTY_QUERY)
    if len(query) > 2000:
        raise McpRequestError(_JSONRPC_INVALID_REQUEST, _MSG_LONG_QUERY)
    session_id = arguments.get("session_id") or str(uuid.uuid4())
    if not isinstance(session_id, str):
        raise McpRequestError(_JSONRPC_INVALID_REQUEST, _MSG_BAD_SESSION)
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise McpRequestError(_JSONRPC_INVALID_REQUEST, _MSG_BAD_SESSION) from None
    return query, session_id


def _text_block(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


async def _handle_tools_call(params: Dict[str, Any]) -> Dict[str, Any]:
    query, session_id = _extract_query_args(params.get("arguments"))
    try:
        result = await _run_axiom_query(query, session_id)
    except PipelineUnavailableError as exc:
        # Sanitized by construction; detail was already logged where raised.
        logger.error("axiom_query unavailable for session %s: %s", session_id, exc)
        return {"content": [_text_block(str(exc))], "isError": True}
    except Exception:
        # Fail-closed envelope: the client learns the query failed and why
        # class-wide, never the raw exception text (model names, hosts,
        # credentials must not cross the MCP boundary). Full detail: stderr.
        logger.exception("axiom_query failed for session %s", session_id)
        return {
            "content": [_text_block(_MSG_QUERY_FAILED)],
            "isError": True,
        }
    return {
        "content": [
            _text_block(result["answer"]),
            _text_block(json.dumps({"sources": result["sources"]}, ensure_ascii=False)),
        ],
        "isError": False,
    }


def _handle_ping(params: Dict[str, Any]) -> Dict[str, Any]:
    return {}  # MCP liveness probe — empty result


_METHODS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
    "ping": _handle_ping,
}
_NOTIFICATIONS = {"notifications/initialized"}


# ---------------------------------------------------------------------------
# stdio loop
# ---------------------------------------------------------------------------


def _response(msg_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


async def _dispatch(line: str, write: Any) -> None:
    """Handle one stdin line; always emit at most one response line."""
    try:
        message = json.loads(line)
        if not isinstance(message, dict):
            raise ValueError("not an object")
    except ValueError:
        await write(_error(None, _JSONRPC_PARSE_ERROR, "Malformed JSON"))
        return

    method = message.get("method")
    msg_id = message.get("id")

    if method in _NOTIFICATIONS:
        return  # notifications are never answered
    if not isinstance(method, str):
        await write(_error(msg_id, _JSONRPC_INVALID_REQUEST, "Missing method"))
        return

    handler = _METHODS.get(method)
    if handler is None:
        await write(_error(msg_id, _JSONRPC_METHOD_NOT_FOUND, f"Unknown method: {method}"))
        return

    params = message.get("params") or {}
    if not isinstance(params, dict):
        await write(_error(msg_id, _JSONRPC_INVALID_REQUEST, "params must be an object"))
        return

    try:
        result = handler(params)
        if asyncio.iscoroutine(result):
            result = await result
        await write(_response(msg_id, result))
    except McpRequestError as exc:
        await write(_error(msg_id, exc.code, exc.message))
    except Exception:
        logger.exception("Unhandled error in %s", method)
        await write(_error(msg_id, _JSONRPC_INTERNAL_ERROR, "Internal error"))


async def _stdout_write(payload: Dict[str, Any]) -> None:
    """Write one protocol frame to stdout. Logs land on stderr only."""
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


async def serve(readline: Any = None, write: Any = None) -> None:
    """Sequential stdio loop: one event loop for the process lifetime (async
    pipeline singletons bind to it once), stdin lines read via executor."""
    loop = asyncio.get_running_loop()
    read = readline if readline is not None else _blocking_readline
    write = write if write is not None else _stdout_write
    logger.info("AXIOM MCP stdio server ready (no network listener)")
    while True:
        line = await loop.run_in_executor(None, read)
        if not line:  # EOF — host closed the pipe
            break
        await _dispatch(line, write)


def _blocking_readline() -> str:
    return sys.stdin.readline()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,  # stdout is protocol-only
    )
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("AXIOM MCP server stopped")


if __name__ == "__main__":
    main()
