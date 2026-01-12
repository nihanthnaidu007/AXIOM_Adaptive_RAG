"""LangSmith observability for AXIOM.

Traces every node execution with custom evaluator tags.
Surfaces trace URL in API response.
"""

import os
import logging
from typing import Optional, Dict, Any

from axiom.config import get_config

logger = logging.getLogger(__name__)


class LangSmithTracer:
    """Manages LangSmith tracing configuration and trace URL generation."""

    def __init__(self):
        self._enabled = False

    def configure(self) -> bool:
        """Check env vars and activate LangSmith tracing if available.

        Returns True if tracing was successfully configured, False otherwise.
        """
        config = get_config()
        tracing_flag = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true" or config.langchain_tracing_v2
        has_key = bool(os.environ.get("LANGCHAIN_API_KEY") or config.langchain_api_key)

        if tracing_flag and has_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = config.langchain_project
            if config.langchain_api_key and not os.environ.get("LANGCHAIN_API_KEY"):
                os.environ["LANGCHAIN_API_KEY"] = config.langchain_api_key
            self._enabled = True
            logger.info("LangSmith tracing enabled — project: %s", config.langchain_project)
            return True

        self._enabled = False
        return False

    def get_run_config(self, run_name: str, session_id: str, metadata: dict | None = None) -> dict:
        """Return a LangChain RunnableConfig dict for graph invocation."""
        metadata = metadata or {}
        return {
            "run_name": run_name,
            "tags": ["axiom", f"session:{session_id}"],
            "metadata": {
                "session_id": session_id,
                "retrieval_strategy": metadata.get("retrieval_strategy", "unknown"),
                "stub_mode": metadata.get("stub_mode", False),
                **metadata,
            },
            "callbacks": [],
        }

    def get_trace_url(self, run_id: str) -> str | None:
        """Return the Smith UI URL for a given run/session ID, or None if tracing is off."""
        if self._enabled:
            return f"https://smith.langchain.com/runs/{run_id}"
        return None

    def is_enabled(self) -> bool:
        return self._enabled


langsmith_tracer = LangSmithTracer()
langsmith_tracer.configure()
