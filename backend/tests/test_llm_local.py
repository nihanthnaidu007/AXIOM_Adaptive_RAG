"""Wave 3 local-generation transport tests (spec D1.5).

httpx-level mocked transports only — no Ollama daemon is contacted. Verified
here: num_predict mapping, transient retry before the first streamed token,
no mid-stream retry, format:json for chat_json, usage reporting, the preserved
cloud path, and the module-level seam wrappers.
"""

import json

import httpx
import pytest

import axiom.llm.client as client_module
from axiom.config import AxiomConfig
from axiom.llm.client import (
    GenerationTransportError,
    LLMClient,
    chat,
    chat_json,
    chat_stream,
)

_LOCAL_CFG = AxiomConfig(
    llm_provider="local",
    embedding_provider="local",
    ollama_host="http://ollama.test",
    anthropic_api_key="",
    openai_api_key="",
    postgres_url="postgresql://t",
)

_CLOUD_CFG = AxiomConfig(
    anthropic_api_key="sk-ant-x",
    openai_api_key="sk-x",
    postgres_url="postgresql://t",
)


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    """Retry backoff collapses to zero in tests; attempts still counted."""
    monkeypatch.setattr(client_module.LLMClient, "_retry_delay", staticmethod(lambda attempt: 0.0))


def _local_client(handler) -> LLMClient:
    return LLMClient(transport=httpx.MockTransport(handler), config=_LOCAL_CFG)


class TestLocalChat:
    @pytest.mark.asyncio
    async def test_maps_max_tokens_to_num_predict(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "hi there"},
                    "prompt_eval_count": 3,
                    "eval_count": 4,
                },
            )

        out = await _local_client(handler).chat("prompt text", max_tokens=123)
        assert out == "hi there"
        body = seen["body"]
        assert body["model"] == "llama3.1:8b"
        assert body["options"]["num_predict"] == 123
        assert body["stream"] is False

    @pytest.mark.asyncio
    async def test_transient_5xx_retried_then_succeeds(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(503, json={"error": "loading"})
            return httpx.Response(200, json={"message": {"content": "ok"}})

        assert await _local_client(handler).chat("p") == "ok"
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_non_transient_4xx_raises_immediately(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(404, json={"error": "model not found"})

        client = _local_client(handler)
        with pytest.raises(GenerationTransportError, match="404"):
            await client.chat("p")
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_transport_error_retries_then_raises(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            raise httpx.ConnectError("connection refused", request=request)

        client = _local_client(handler)
        with pytest.raises(GenerationTransportError):
            await client.chat("p")
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_reports_ollama_usage_fields(self, monkeypatch):
        recorded: dict = {}

        def fake_usage(prompt_tokens, completion_tokens):
            recorded.update(prompt=prompt_tokens, completion=completion_tokens)

        monkeypatch.setattr(client_module, "record_llm_usage", fake_usage)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "message": {"content": "x"},
                    "prompt_eval_count": 11,
                    "eval_count": 22,
                },
            )

        await _local_client(handler).chat("p")
        assert recorded == {"prompt": 11, "completion": 22}


class TestLocalStream:
    @staticmethod
    def _ndjson(*frames: dict) -> bytes:
        return ("\n".join(json.dumps(f) for f in frames) + "\n").encode()

    @pytest.mark.asyncio
    async def test_deltas_stream_in_order(self, monkeypatch):
        recorded: dict = {}
        monkeypatch.setattr(
            client_module,
            "record_llm_usage",
            lambda p, c: recorded.update(prompt=p, completion=c),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=self._ndjson(
                    {"message": {"content": "Hello"}, "done": False},
                    {"message": {"content": " world"}, "done": False},
                    {
                        "message": {"content": ""},
                        "done": True,
                        "prompt_eval_count": 5,
                        "eval_count": 7,
                    },
                ),
            )

        client = _local_client(handler)
        deltas = [d async for d in client.chat_stream("p", max_tokens=64)]
        assert "".join(deltas) == "Hello world"
        assert recorded == {"prompt": 5, "completion": 7}

    @pytest.mark.asyncio
    async def test_retries_before_first_token(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(
                200,
                content=self._ndjson(
                    {"message": {"content": "Hi"}, "done": True},
                ),
            )

        client = _local_client(handler)
        deltas = [d async for d in client.chat_stream("p")]
        assert deltas == ["Hi"]
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_no_retry_after_emission(self):
        """Mid-stream failure propagates — a retry would duplicate text."""
        calls = {"n": 0}

        async def failing_stream():
            yield b'{"message": {"content": "partial"}, "done": false}\n'
            raise httpx.ReadError("connection dropped mid-stream")

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, content=failing_stream())

        client = _local_client(handler)
        emitted = []
        with pytest.raises(httpx.ReadError):
            async for delta in client.chat_stream("p"):
                emitted.append(delta)
        assert emitted == ["partial"]
        assert calls["n"] == 1


class TestLocalChatJson:
    @pytest.mark.asyncio
    async def test_sends_format_json_and_parses(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200, json={"message": {"content": '{"verdict": "pass", "score": 4}'}}
            )

        client = _local_client(handler)
        result = await client.chat_json("p")
        assert result == {"verdict": "pass", "score": 4}
        assert seen["body"]["format"] == "json"

    @pytest.mark.asyncio
    async def test_malformed_json_raises_value_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"message": {"content": "definitely not json"}})

        client = _local_client(handler)
        with pytest.raises(ValueError):
            await client.chat_json("p")


class TestProbeAndLifecycle:
    @pytest.mark.asyncio
    async def test_probe_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert request.url.path == "/api/generate"
            assert body["options"]["num_predict"] == 1
            return httpx.Response(200, json={"response": "pong"})

        assert await _local_client(handler).probe() is True

    @pytest.mark.asyncio
    async def test_probe_failure_returns_false(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        assert await _local_client(handler).probe() is False

    @pytest.mark.asyncio
    async def test_aclose_local_closes_httpx_client(self):
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            return httpx.Response(200, json={})

        client = _local_client(handler)
        inner = client._client
        await client.aclose()
        assert inner.is_closed


class TestCloudPathPreserved:
    @pytest.mark.asyncio
    async def test_cloud_chat_still_works_via_seam(self, monkeypatch):
        class _Text:
            text = "cloud answer"

        class _Usage:
            input_tokens = 1
            output_tokens = 2

        class _Response:
            content = [_Text()]
            usage = _Usage()

        class _Messages:
            async def create(self, **kwargs):
                return _Response()

        class _StubAnthropic:
            def __init__(self, **kwargs):
                pass

            messages = _Messages()

            def close(self):
                pass

        monkeypatch.setattr(client_module, "AsyncAnthropic", _StubAnthropic)
        client = LLMClient(config=_CLOUD_CFG)
        assert await client.chat("p") == "cloud answer"


class TestModuleLevelSeam:
    """Call sites import module-level chat/chat_stream/chat_json — the seam
    contract (zero call-site edits) keeps those wrappers authoritative."""

    @pytest.mark.asyncio
    async def test_chat_wrapper_delegates_to_singleton(self, monkeypatch):
        async def fake_chat(prompt, model=None, max_tokens=2000):
            return "delegated"

        monkeypatch.setattr(client_module.llm_client, "chat", fake_chat)
        assert await chat("p") == "delegated"

    @pytest.mark.asyncio
    async def test_chat_stream_wrapper_returns_iterator(self, monkeypatch):
        async def fake_stream(prompt, model=None, max_tokens=2000):
            yield "d1"

        monkeypatch.setattr(client_module.llm_client, "chat_stream", fake_stream)
        deltas = [d async for d in chat_stream("p")]
        assert deltas == ["d1"]

    @pytest.mark.asyncio
    async def test_chat_json_wrapper_delegates(self, monkeypatch):
        async def fake_json(prompt, model=None, max_tokens=2000):
            return {"k": "v"}

        monkeypatch.setattr(client_module.llm_client, "chat_json", fake_json)
        assert await chat_json("p") == {"k": "v"}
