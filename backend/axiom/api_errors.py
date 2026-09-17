"""Structured error envelopes for API failure paths.

Unexpected exceptions must never reach clients as raw ``str(exc)`` — that can
leak internals (DSNs, SQL, file paths). Instead every failure response uses a
stable envelope shape:

HTTP (inside FastAPI's ``detail`` convention, so ``detail.error`` stays a
string for existing consumers):

    {"detail": {"error": "<safe message>", "code": "<machine code>",
                "context": {"<safe ids>": ...}}}

SSE error events:

    {"type": "error", "code": "<machine code>", "message": "<safe message>"}

Full exception detail stays server-side in the logs.
"""

from typing import Any, Dict, Optional

# Stable, client-meaningful failure codes.
INTERNAL_ERROR = "internal_error"

# Safe, non-leaking message used for any unexpected server-side exception.
GENERIC_INTERNAL_MESSAGE = (
    "An unexpected error occurred while processing the request. "
    "Check server logs for details."
)

# Markers frequently found in exception text that indicate internal detail.
_INTERNAL_MARKERS = (
    "postgresql",
    "postgres://",
    "asyncpg",
    "redis",
    "sqlalchemy",
    "traceback",
    "file \"",
    "0x",
    "password",
    "localhost",
    "127.0.0.1",
    ".env",
    "api_key",
    "/home/",
    "/usr/",
    "/tmp/",
)


def error_detail(
    code: str = INTERNAL_ERROR,
    message: str = GENERIC_INTERNAL_MESSAGE,
    **context: Any,
) -> Dict[str, Any]:
    """Build the HTTP error envelope passed as ``HTTPException(detail=...)``.

    ``detail.error`` remains a plain string (the frontend reads it directly);
    ``code`` is a stable machine-readable identifier and ``context`` carries
    only caller-supplied, non-sensitive identifiers (session ids, filenames).
    """
    envelope: Dict[str, Any] = {"error": message, "code": code}
    if context:
        envelope["context"] = context
    return envelope


def sse_error_event(
    code: str = INTERNAL_ERROR,
    message: str = GENERIC_INTERNAL_MESSAGE,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the SSE error event payload (``{"type": "error", ...}``).

    ``request_id`` is included when known so a failed stream can be
    correlated with the server's JSON logs.
    """
    event: Dict[str, Any] = {"type": "error", "code": code, "message": message}
    if request_id:
        event["request_id"] = request_id
    return event


def is_safe_message(message: Optional[str]) -> bool:
    """Return True when a message carries no recognizable internal detail.

    Used defensively before echoing any non-generic message to a client.
    """
    if not message:
        return True
    lowered = message.lower()
    return not any(marker in lowered for marker in _INTERNAL_MARKERS)
