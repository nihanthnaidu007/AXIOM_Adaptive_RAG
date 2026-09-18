#!/usr/bin/env python3
"""Scripted MCP stdio host — dogfoods backend/mcp_server.py end to end.

Spawns the real server as a subprocess and speaks newline-delimited JSON-RPC
over real pipes: initialize -> notifications/initialized -> tools/list ->
tools/call, printing every exchanged frame to stdout as a labeled transcript.

Two scenarios run back to back:

1. real-server — the actual pipeline runs. On a host without
   Redis/pgvector/Ollama the tool answers with its sanitized fail-closed
   envelope; that envelope is the correct live behavior, not a defect, and
   the transcript shows the raw failure detail stayed on the server's stderr.
2. stub-bridge — the identical protocol path with ONLY the pipeline bridge
   stubbed (same seam the tests patch), so service-less hosts can still
   capture the answer + sources success transcript. The server's dispatch
   loop, envelopes, and JSON-RPC framing are the real ones.

Usage (from backend/):
    python scripts/dogfood_mcp.py [--timeout SEC]

Zero dependencies beyond the standard library, like the server it exercises.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from typing import Any, Dict, List, Optional

STUB_RUNNER = '''"""Server-side runner: real mcp_server with a stubbed pipeline bridge."""
import sys

sys.path.insert(0, {backend_dir!r})

import mcp_server


async def _stub(query, session_id):
    return {{
        "answer": "Fully-local mode answers via Ollama; retrieval hit 3 chunks.",
        "session_id": session_id,
        "sources": [
            {{"kind": "document", "source": "docs/architecture.md", "chunk_id": "doc-1",
              "score": 0.87, "snippet": "AXIOM combines BM25, vector, and hybrid retrieval."}},
            {{"kind": "document", "source": "README.md", "chunk_id": "doc-2",
              "score": 0.71, "snippet": "Ollama runs the generator and embeddings locally."}},
            {{"kind": "web", "source": "https://ollama.com/library", "chunk_id": "",
              "score": 0.4, "snippet": "Ollama model library."}},
        ],
    }}


mcp_server._run_axiom_query = _stub
mcp_server.main()
'''


def _frame(method: str, msg_id: Optional[int] = None, params: Optional[Dict[str, Any]] = None) -> str:
    message: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if msg_id is not None:
        message["id"] = msg_id
    if params is not None:
        message["params"] = params
    return json.dumps(message)


def _drive(label: str, cmd: list, cwd: str, timeout: float) -> None:
    """Run the four-step handshake against one server process; print frames."""
    print(f"\n=== AXIOM MCP dogfood — {label} ===")
    print(f"    command: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    watchdog = threading.Timer(timeout, proc.kill)
    watchdog.start()
    stderr_lines: List[str] = []
    try:
        stderr_stream = proc.stderr
        assert stderr_stream is not None  # stderr=PIPE was requested
        stderr_thread = threading.Thread(
            target=lambda: stderr_lines.extend(iter(stderr_stream.readline, ""))
        )
        stderr_thread.start()

        def exchange(frame: str, expect_reply: bool = True) -> None:
            assert proc.stdin is not None and proc.stdout is not None
            print(f"--> {frame}")
            proc.stdin.write(frame + "\n")
            proc.stdin.flush()
            if expect_reply:
                reply = proc.stdout.readline().rstrip("\n")
                print(f"<-- {reply}")

        # MCP handshake: initialize -> (notification, never answered) -> ready
        exchange(_frame("initialize", msg_id=1, params={"protocolVersion": "2024-11-05"}))
        exchange(_frame("notifications/initialized"), expect_reply=False)
        exchange(_frame("tools/list", msg_id=2))
        exchange(
            _frame(
                "tools/call",
                msg_id=3,
                params={
                    "name": "axiom_query",
                    "arguments": {"query": "How does AXIOM retrieve documents?"},
                },
            )
        )
    finally:
        watchdog.cancel()
        if proc.stdin is not None:
            proc.stdin.close()  # EOF ends the serve loop cleanly
        proc.wait(timeout=10)
        print(f"    server exit code: {proc.returncode}")
        captured = "".join(stderr_lines)
        print("    server stderr (raw failures must appear HERE, never in responses):")
        for line in captured.strip().splitlines()[-4:]:
            print(f"      | {line}")
        if proc.poll() is None:
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dogfood the AXIOM MCP stdio server.")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-scenario watchdog seconds")
    args = parser.parse_args()

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.NamedTemporaryFile("w", suffix="_mcp_stub_runner.py", delete=False) as fh:
        fh.write(STUB_RUNNER.format(backend_dir=backend_dir))
        stub_path = fh.name

    try:
        # Scenario 1: the untouched server — real pipeline, fail-closed live.
        _drive(
            "real server (pipeline live; services absent here -> sanitized error expected)",
            [sys.executable, "mcp_server.py"],
            backend_dir,
            args.timeout,
        )
        # Scenario 2: real protocol, stubbed bridge -> answer + sources.
        _drive(
            "stubbed pipeline bridge (answer + sources transcript)",
            [sys.executable, stub_path],
            backend_dir,
            args.timeout,
        )
    finally:
        os.unlink(stub_path)


if __name__ == "__main__":
    main()
