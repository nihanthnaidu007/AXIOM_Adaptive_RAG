"""Wave 4 S3 connector tests (D4) — no network.

Covers:
- extension allowlist + prefix filtering + recursive/top-level mode
- per-object size cap is fail-visible (error row, fetch skipped)
- per-object GetObject failure -> sanitized error row, siblings still fetched
- bucket-level listing failure -> S3ConnectorError with sanitized reason
- stable s3:// bucket/key source keys and S3FetchedObject metadata
- one botocore Stubber round trip proving the paginator call shape
"""

import botocore.session
import pytest
from botocore.stub import Stubber

from axiom.connectors.s3_connector import (
    S3_ALLOWED_EXTENSIONS,
    S3ConnectorError,
    _extension_of,
    _list_allowed_keys,
    fetch_s3_objects,
)


class _FakeCfg:
    """Config stand-in exposing only the S3 knobs the connector reads."""

    def __init__(self, **overrides):
        self.s3_bucket = "axiom-test"
        self.s3_prefix = ""
        self.s3_recursive = True
        self.s3_max_object_size_mb = 25
        self.s3_region = "us-east-1"
        self.s3_access_key_id = "test-key"
        self.s3_secret_access_key = "test-secret"
        self.s3_endpoint_url = ""
        for k, v in overrides.items():
            setattr(self, k, v)


class _FakeBody:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _StubbedClient:
    """Paginator/get_object double that honors Prefix and records call kwargs."""

    def __init__(self, pages, get_object=None):
        self._pages = pages
        self._get_object = get_object or {}
        self.paginate_calls = []

    def get_paginator(self, _op):
        return self

    def paginate(self, **kwargs):
        self.paginate_calls.append(kwargs)
        prefix = kwargs.get("Prefix", "")
        pages = []
        for page in self._pages:
            contents = [o for o in page.get("Contents", []) if o.get("Key", "").startswith(prefix)]
            pages.append({"Contents": contents})
        return pages

    def get_object(self, Bucket, Key):
        entry = self._get_object[Key]
        if isinstance(entry, Exception):
            raise entry
        return {"Body": _FakeBody(entry)}


def test_extension_allowlist_filters_listing():
    client = _StubbedClient(
        [
            {
                "Contents": [
                    {"Key": "reports/q1.pdf", "Size": 100},
                    {"Key": "reports/q1.eps", "Size": 100},  # not in the allowlist
                    {"Key": "reports/notes.txt", "Size": 10},
                    {"Key": "reports/empty/", "Size": 0},  # directory marker
                ]
            }
        ]
    )

    listed = _list_allowed_keys(client, _FakeCfg(s3_prefix="reports/"))

    assert [item["key"] for item in listed] == ["reports/q1.pdf", "reports/notes.txt"]


def test_prefix_is_passed_to_listing():
    client = _StubbedClient([{"Contents": [{"Key": "reports/q1.pdf", "Size": 100}]}])

    _list_allowed_keys(client, _FakeCfg(s3_prefix="reports/"))

    assert client.paginate_calls == [{"Bucket": "axiom-test", "Prefix": "reports/"}]


def test_non_recursive_mode_sets_delimiter():
    client = _StubbedClient([{"Contents": [{"Key": "top.txt", "Size": 4}]}])

    _list_allowed_keys(client, _FakeCfg(s3_recursive=False))

    assert client.paginate_calls[0]["Delimiter"] == "/"


def test_extension_of_normalizes_case():
    assert _extension_of("a/b/report.PDF") == ".pdf"
    assert _extension_of("noext") == ""


@pytest.mark.asyncio
async def test_fetch_skips_oversize_objects_with_visible_error(monkeypatch):
    """The 25 MiB default cap: oversize object -> error row, small one fetched."""
    from axiom.connectors import s3_connector as s3mod

    client = _StubbedClient(
        [{"Contents": [
            {"Key": "docs/a.txt", "Size": 5},
            {"Key": "docs/big.txt", "Size": 26 * 1024 * 1024},
        ]}],
        get_object={"docs/a.txt": b"hello"},
    )
    monkeypatch.setattr(s3mod, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(s3mod, "_build_client", lambda cfg: client)

    fetched, errors = await fetch_s3_objects()

    assert [f.source for f in fetched] == ["s3://axiom-test/docs/a.txt"]
    assert fetched[0].content == b"hello"
    assert fetched[0].filename == "a.txt"
    assert fetched[0].size == 5
    assert fetched[0].content_hash
    assert fetched[0].fetched_at is not None
    assert len(errors) == 1
    assert errors[0]["source"] == "s3://axiom-test/docs/big.txt"
    assert "cap" in errors[0]["reason"].lower()


@pytest.mark.asyncio
async def test_per_object_failure_is_sanitized_and_non_terminal(monkeypatch):
    from axiom.connectors import s3_connector as s3mod

    class _ExplodingBody:
        def read(self):
            raise TimeoutError("read timed out after 30s host=10.0.0.9")

    class _Client(_StubbedClient):
        def get_object(self, Bucket, Key):
            if Key == "docs/broken.txt":
                return {"Body": _ExplodingBody()}
            return {"Body": _FakeBody(b"fine")}

    client = _Client(
        [{"Contents": [
            {"Key": "docs/broken.txt", "Size": 4},
            {"Key": "docs/ok.txt", "Size": 4},
        ]}],
        get_object={},
    )
    monkeypatch.setattr(s3mod, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(s3mod, "_build_client", lambda cfg: client)

    fetched, errors = await fetch_s3_objects()

    assert [f.source for f in fetched] == ["s3://axiom-test/docs/ok.txt"]
    assert len(errors) == 1
    assert errors[0]["source"] == "s3://axiom-test/docs/broken.txt"
    # Sanitized: the internal host from the exception must not leak into the record.
    assert "10.0.0.9" not in errors[0]["reason"]


@pytest.mark.asyncio
async def test_bucket_level_listing_failure_raises_s3_connector_error(monkeypatch):
    from axiom.connectors import s3_connector as s3mod

    class _BrokenClient:
        def get_paginator(self, _op):
            raise PermissionError("denied by policy for principal arn:aws:iam::123:user/x")

    monkeypatch.setattr(s3mod, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(s3mod, "_build_client", lambda cfg: _BrokenClient())

    with pytest.raises(S3ConnectorError) as excinfo:
        await fetch_s3_objects()

    assert "axiom-test" in str(excinfo.value)
    assert "listing failed" in str(excinfo.value).lower()
    # Sanitized: the IAM ARN from the raw botocore error must not surface.
    assert "arn:aws:iam" not in str(excinfo.value)


def test_stubber_round_trip_on_list():
    """Sanity: botocore Stubber is wired correctly for the list_objects_v2 shape."""
    client = botocore.session.get_session().create_client("s3", region_name="us-east-1")
    stubber = Stubber(client)
    stubber.add_response(
        "list_objects_v2",
        {"Contents": [{"Key": "a.txt", "Size": 3}]},
        {"Bucket": "axiom-test", "Prefix": ""},
    )
    with stubber:
        listed = _list_allowed_keys(client, _FakeCfg())
    assert listed == [{"key": "a.txt", "size": 3}]


def test_allowlist_covers_the_parse_seam_formats():
    assert {".txt", ".md", ".pdf", ".html", ".htm", ".csv"} <= S3_ALLOWED_EXTENSIONS
