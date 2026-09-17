"""Content streaming bridge between graph nodes and the SSE endpoint.

The LangGraph nodes communicate through AxiomState only — there is no channel
for token-level output. This module adds one without changing the state
contract: the /query/stream endpoint installs a ``ContentSink`` in a
contextvar before the graph runs; generation nodes look it up and publish
answer deltas into it as the LLM streams.

The sink object is shared by reference: even when LangGraph runs nodes in
sub-tasks (which copy the context at creation), every holder publishes into
the same queue the endpoint generator consumes.

Non-streaming requests install no sink — ``get_content_sink()`` returns None
and the nodes take the exact one-shot ``chat()`` path they always did.
"""

import asyncio
from contextvars import ContextVar
from typing import AsyncIterator, Optional

from axiom.llm.client import chat, chat_stream

# Frame kinds carried on the sink queue. ("content", delta) is generation
# text; the endpoint's own producer task pushes ("node", frame) for graph
# trace events and the terminal ("final_state", ...) / ("graph_error", ...).
Frame = tuple[str, object]

_sink_var: ContextVar[Optional["ContentSink"]] = ContextVar(
    "axiom_content_sink", default=None
)


def get_content_sink() -> Optional["ContentSink"]:
    """The active sink for this request, or None outside streaming mode."""
    return _sink_var.get()


def install_content_sink(sink: "ContentSink") -> object:
    """Activate ``sink`` for the current context; returns the reset token."""
    return _sink_var.set(sink)


def reset_content_sink(token: object) -> None:
    """Restore the previous contextvar state captured by ``install``."""
    _sink_var.reset(token)  # type: ignore[arg-type]


class ContentSink:
    """Queues generation deltas (and endpoint control frames) for the SSE loop."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[Frame] = asyncio.Queue()
        # Downstream multiplexer queues (e.g. the SSE endpoint's frame queue)
        # that receive a copy of every frame in publish order.
        self.subscribers: list[asyncio.Queue] = []
        self.emitted_any = False

    def subscribe(self, queue: asyncio.Queue) -> None:
        """Forward every subsequently published frame into ``queue``."""
        self.subscribers.append(queue)

    def publish(self, delta: str) -> None:
        """Queue one text delta. No-op after close (client already gone)."""
        if not delta:
            return
        frame: Frame = ("content", delta)
        self.queue.put_nowait(frame)
        for q in self.subscribers:
            q.put_nowait(frame)
        self.emitted_any = True

    def publish_text(self, text: str, chunk_size: int = 64) -> None:
        """Publish a complete text as a series of deltas.

        Used where the answer is already known in full (semantic-cache hits)
        so cache hits stream through the same ``content`` events.
        """
        for i in range(0, len(text), chunk_size):
            self.publish(text[i : i + chunk_size])


async def stream_answer(
    prompt: str,
    sink: ContentSink,
    max_tokens: int = 2000,
) -> str:
    """Generate ``prompt`` while streaming deltas into ``sink``.

    Returns the full accumulated answer (the same text the non-streaming
    path would have returned). Raises on generation failure — callers keep
    their existing error handling.
    """
    parts: list[str] = []
    async for delta in chat_stream(prompt, max_tokens=max_tokens):
        parts.append(delta)
        sink.publish(delta)
    return "".join(parts)


async def generate_with_optional_streaming(
    prompt: str,
    max_tokens: int = 2000,
) -> str:
    """One-shot chat when no sink is installed; streaming chat when one is."""
    sink = get_content_sink()
    if sink is None:
        return await chat(prompt, max_tokens=max_tokens)
    return await stream_answer(prompt, sink, max_tokens=max_tokens)


async def drain_until_close(sink: ContentSink) -> AsyncIterator[Frame]:
    """Yield frames until the sink is closed (utility for tests/tools)."""
    while True:
        yield await sink.queue.get()
