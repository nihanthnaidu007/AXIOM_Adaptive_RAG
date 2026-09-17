"""Unit tests for the structured error envelope helpers."""

from axiom.api_errors import (
    GENERIC_INTERNAL_MESSAGE,
    INTERNAL_ERROR,
    error_detail,
    is_safe_message,
    sse_error_event,
)


class TestErrorDetail:
    def test_envelope_shape(self):
        detail = error_detail()
        assert detail == {"error": GENERIC_INTERNAL_MESSAGE, "code": "internal_error"}

    def test_envelope_carries_context(self):
        detail = error_detail(INTERNAL_ERROR, "msg", session_id="s1", node="classify")
        assert detail["error"] == "msg"
        assert detail["code"] == "internal_error"
        assert detail["context"] == {"session_id": "s1", "node": "classify"}

    def test_no_context_key_when_empty(self):
        assert "context" not in error_detail()

    def test_error_is_always_a_string(self):
        """The frontend reads detail.error directly — it must stay a string."""
        assert isinstance(error_detail()["error"], str)


class TestSSEErrorEvent:
    def test_event_shape(self):
        event = sse_error_event()
        assert event == {
            "type": "error",
            "code": "internal_error",
            "message": GENERIC_INTERNAL_MESSAGE,
        }

    def test_custom_code(self):
        assert sse_error_event("query_timeout", "timed out")["code"] == "query_timeout"


class TestIsSafeMessage:
    def test_detects_connection_strings(self):
        assert not is_safe_message("asyncpg failed for postgresql://user:pass@host")

    def test_detects_paths_and_addresses(self):
        assert not is_safe_message("cannot open /home/user/.env")
        assert not is_safe_message("refused 127.0.0.1:5432")

    def test_detects_secrets(self):
        assert not is_safe_message("invalid api_key: sk-123")

    def test_plain_messages_pass(self):
        assert is_safe_message("Query timed out. Try a simpler query.")
        assert is_safe_message(None)
        assert is_safe_message("")
