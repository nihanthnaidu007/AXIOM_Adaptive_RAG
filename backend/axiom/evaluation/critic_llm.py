"""AXIOM Critic LLM - Ollama client for RAGAS evaluation."""

import logging
import time

import httpx

from axiom.config import get_config

logger = logging.getLogger(__name__)

_ollama_status_cache = {"connected": None, "checked_at": 0}


class CriticLLM:
    def __init__(self):
        cfg = get_config()
        self._host = cfg.ollama_host
        self._model = cfg.ollama_critic_model
        self._connected = False
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))

    async def connect(self) -> bool:
        try:
            resp = await self._client.post(
                f"{self._host}/api/generate",
                json={"model": self._model, "prompt": "Say OK.", "stream": False,
                      "options": {"num_predict": 5}},
                timeout=10.0,
            )
            if resp.status_code == 200 and resp.json().get("response"):
                self._connected = True
                return True
            self._connected = False
            return False
        except Exception as exc:
            logger.error("Ollama connection failed: %s", exc)
            self._connected = False
            return False

    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        if not self._connected:
            return ""
        try:
            resp = await self._client.post(
                f"{self._host}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False,
                      "options": {"num_predict": max_tokens}},
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
            logger.error("Ollama generate HTTP %d", resp.status_code)
            return ""
        except Exception as exc:
            logger.error("Ollama generate failed: %s", exc)
            return ""

    async def is_connected(self) -> bool:
        """Connectivity check with 10-second TTL cache to avoid pinging Ollama on every query."""
        now = time.monotonic()
        if _ollama_status_cache["connected"] is not None and now - _ollama_status_cache["checked_at"] < 10:
            return _ollama_status_cache["connected"]
        try:
            resp = await self._client.post(
                f"{self._host}/api/generate",
                json={"model": self._model, "prompt": "ping", "stream": False,
                      "options": {"num_predict": 1}},
                timeout=3.0,
            )
            self._connected = resp.status_code == 200
        except Exception:
            self._connected = False
        _ollama_status_cache["connected"] = self._connected
        _ollama_status_cache["checked_at"] = now
        return self._connected


critic_llm = CriticLLM()
