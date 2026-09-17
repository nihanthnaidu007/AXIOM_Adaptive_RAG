"""Structured JSON application logging with per-request IDs.

``LOG_FORMAT=json`` (default) emits one JSON object per log line;
``LOG_FORMAT=text`` keeps the legacy human-readable format for local dev.

Every record carries ``request_id`` — taken from the record's ``extra``
when the caller passed one, otherwise from the request-scoped ContextVar
that server.py's request-context middleware populates — so any log line can
be correlated back to the API call that produced it. Importing this module
must stay cheap and side-effect-light: ``configure_logging`` is called once
from server.py, after ``.env`` is loaded (so LOG_FORMAT/LOG_LEVEL apply).
"""

import json
import logging
import os
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict

# Set by the request-context middleware; empty outside a request.
request_id_var: ContextVar[str] = ContextVar("axiom_request_id", default="")

# LogRecord attributes that are either standard or derived; anything else in
# record.__dict__ came from ``extra=`` (or a filter) and belongs in the JSON
# payload.
_RESERVED_ATTRS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "module", "msecs", "message",
    "msg", "name", "pathname", "process", "processName", "relativeCreated",
    "stack_info", "stacklevel", "thread", "threadName", "taskName",
})

_TEXT_FORMAT = "%(levelname)s: %(name)s — %(message)s"


def get_request_id() -> str:
    """Request ID for the current context ("" when outside a request)."""
    return request_id_var.get()


def normalize_request_id(raw: str) -> str:
    """Validate a client-supplied X-Request-ID; "" when unusable.

    Accepts printable ASCII up to 128 chars so the value is safe to echo
    into response headers and JSON log lines. Validation runs on the whole
    value with no stripping: control characters (including \x1f, which
    str.strip would silently remove as Unicode whitespace) mean the value
    is unusable and gets replaced by a server-issued ID.
    """
    value = raw or ""
    if not value or len(value) > 128:
        return ""
    if not all(0x20 < ord(ch) < 0x7F for ch in value):
        return ""
    return value


class JsonLogFormatter(logging.Formatter):
    """Render each LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = getattr(record, "request_id", "") or request_id_var.get()
        if request_id:
            payload["request_id"] = request_id

        # Merge caller-supplied ``extra=`` fields, skipping logging internals.
        for key, value in record.__dict__.items():
            if key in _RESERVED_ATTRS or key in payload or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Install the JSON (or text) root handler once.

    No-op when a root handler already exists (pytest and uvicorn configure
    their own root logging; clobbering it would break log capture).
    """
    root = logging.getLogger()
    if root.handlers:
        return

    log_format = os.environ.get("LOG_FORMAT", "json").strip().lower()
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonLogFormatter() if log_format != "text" else logging.Formatter(_TEXT_FORMAT)
    )
    root.addHandler(handler)
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
