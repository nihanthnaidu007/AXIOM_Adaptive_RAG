"""Wave 4 web-crawl connector tests (D4) — httpx MockTransport, no network.

Covers:
- BFS with same-host link following; cross-host links recorded but not followed
- robots.txt honored by default; unreachable robots fails open
- connector-owned Content-Type allowlist (R11: the HTTP endpoint's MIME
  sniffing does not apply here)
- byte cap, max-pages cap, per-host rate limiting, fetch-time dedup
- CrawlFetchedPage metadata: stable URL source key, sha256, extracted title
"""

import asyncio
from datetime import datetime

import httpx
import pytest

from axiom.connectors import crawl_connector as crawlmod
from axiom.connectors.crawl_connector import (
    CrawlConnectorError,
    CrawlDedupStore,
    CrawlFetchedPage,
    _content_type_allowed,
    _HostRateLimiter,
    crawl,
    parse_seeds,
)


class _FakeCfg:
    """Config stand-in exposing only the crawl knobs the connector reads."""

    def __init__(self, **overrides):
        self.crawl_seeds = ""
        self.crawl_max_pages = 25
        self.crawl_max_depth = 2
        self.crawl_rate_limit_seconds = 0.0
        self.crawl_respect_robots = True
        self.crawl_max_page_bytes = 2_000_000
        for k, v in overrides.items():
            setattr(self, k, v)


@pytest.fixture()
def fake_cfg(monkeypatch):
    cfg = _FakeCfg()
    monkeypatch.setattr(crawlmod, "get_config", lambda: cfg)
    return cfg


HTML_INDEX = (
    "<html><head><title>AXIOM Docs</title></head><body>"
    "<h1>Welcome</h1>"
    '<a href="/docs/guide">Guide</a>'
    '<a href="http://elsewhere.example.com/page">External</a>'
    "</body></html>"
)
HTML_GUIDE = "<html><head><title>Guide</title></head><body><p>Installation steps.</p></body></html>"


def _transport(handler):
    return httpx.MockTransport(handler)


def _html(body: str) -> httpx.Response:
    """HTML response with a realistic Content-Type — MockTransport's text= alone
    would send text/plain and the connector (correctly) would not parse it."""
    return httpx.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8"})


def test_happy_path_crawl_follows_same_host_links(fake_cfg):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(404)
        if path == "/":
            return _html(HTML_INDEX)
        if path == "/docs/guide":
            return _html(HTML_GUIDE)
        return httpx.Response(404)

    pages, errors = asyncio.run(crawl(["http://testserver/"], transport=_transport(handler)))

    assert errors == []
    assert [p.source for p in pages] == ["http://testserver/", "http://testserver/docs/guide"]
    # Cross-host link from the index page: recorded as a link but never fetched.
    assert not any("elsewhere" in r for r in requests)


def test_crawl_page_metadata_is_complete(fake_cfg):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return _html(HTML_GUIDE)

    pages, _ = asyncio.run(crawl(["http://testserver/docs"], transport=_transport(handler)))

    page: CrawlFetchedPage = pages[0]
    assert page.source == "http://testserver/docs"
    assert page.title == "Guide"
    assert b"Installation steps." in page.content
    assert page.ext == ".txt"
    assert page.size == len(page.content)
    assert page.content_hash
    assert isinstance(page.fetched_at, datetime)


def test_robots_disallow_blocks_fetch(fake_cfg):
    fetched = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        fetched.append(str(request.url))
        return httpx.Response(200, text="<html><body>secret</body></html>")

    pages, errors = asyncio.run(
        crawl(["http://testserver/private/x"], transport=_transport(handler))
    )

    assert pages == []
    assert fetched == [], "disallowed URL must never be fetched"
    assert len(errors) == 1
    assert "robots" in errors[0]["reason"].lower()


def test_robots_allowance_lets_fetch_proceed(fake_cfg):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return _html("<html><body>public</body></html>")

    pages, errors = asyncio.run(crawl(["http://testserver/public"], transport=_transport(handler)))

    assert errors == []
    assert len(pages) == 1


def test_content_type_outside_allowlist_is_rejected(fake_cfg):
    """R11: the connector applies its own allowlist — a JSON API response is
    not ingested even though it is text."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, json={"answer": 42})

    pages, errors = asyncio.run(crawl(["http://testserver/api"], transport=_transport(handler)))

    assert pages == []
    assert len(errors) == 1
    assert "allowlist" in errors[0]["reason"]
    assert "application/json" in errors[0]["reason"]


def test_byte_cap_rejects_oversized_page(fake_cfg):
    fake_cfg.crawl_max_page_bytes = 64

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, text="<html>" + "y" * 200 + "</html>")

    pages, errors = asyncio.run(crawl(["http://testserver/big"], transport=_transport(handler)))

    assert pages == []
    assert len(errors) == 1
    assert "byte cap" in errors[0]["reason"]


def test_max_pages_cap_stops_the_crawl(fake_cfg):
    fake_cfg.crawl_max_pages = 1

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return _html(HTML_INDEX)

    pages, _ = asyncio.run(crawl(["http://testserver/"], transport=_transport(handler)))

    assert len(pages) == 1


def test_max_depth_zero_ignores_links(fake_cfg):
    fake_cfg.crawl_max_depth = 0

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return _html(HTML_INDEX)

    pages, _ = asyncio.run(crawl(["http://testserver/"], transport=_transport(handler)))

    assert [p.source for p in pages] == ["http://testserver/"]


def test_fetch_error_becomes_sanitized_error_row(fake_cfg):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(500, text="boom with internal host 10.9.9.9")

    pages, errors = asyncio.run(crawl(["http://testserver/fail"], transport=_transport(handler)))

    assert pages == []
    assert len(errors) == 1
    assert "Fetch failed" in errors[0]["reason"]


@pytest.mark.asyncio
async def test_dedup_store_skips_refetches():
    store = CrawlDedupStore()

    assert await store.seen("http://x/a") is False
    await store.mark_fetched("http://x/a")
    assert await store.seen("http://x/a") is True
    assert await store.seen("http://x/b") is False


def test_second_run_with_shared_dedup_skips_pages(fake_cfg):
    fetch_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        fetch_count["n"] += 1
        return _html("<html><body>content</body></html>")

    dedup = CrawlDedupStore()
    first, _ = asyncio.run(
        crawl(["http://testserver/page"], dedup=dedup, transport=_transport(handler))
    )
    second, second_errors = asyncio.run(
        crawl(["http://testserver/page"], dedup=dedup, transport=_transport(handler))
    )

    assert len(first) == 1
    assert second == []
    assert second_errors == [], "dedup skip is not an error"
    assert fetch_count["n"] == 1


def test_failed_fetch_is_not_marked_for_dedup(fake_cfg):
    """A failed fetch must not poison the dedup cache: the next run retries."""
    fail_first = {"n": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if fail_first["n"]:
            fail_first["n"] = False
            raise httpx.ConnectError("server down", request=request)
        return _html("<html><body>recovered</body></html>")

    dedup = CrawlDedupStore()
    first, first_errors = asyncio.run(
        crawl(["http://testserver/page"], dedup=dedup, transport=_transport(handler))
    )
    second, second_errors = asyncio.run(
        crawl(["http://testserver/page"], dedup=dedup, transport=_transport(handler))
    )

    assert first == []
    assert len(first_errors) == 1
    assert len(second) == 1
    assert second_errors == []
    assert second[0].source == "http://testserver/page"


def test_rate_limiter_waits_between_same_host_requests():
    sleeps = []

    async def _fake_sleep(seconds):
        sleeps.append(round(seconds, 3))

    # Start the clock away from 0 (like time.monotonic): the FIRST request to a
    # host must not wait; only the immediate second one does.
    now = [100.0]
    limiter = _HostRateLimiter(1.5, sleep=_fake_sleep, clock=lambda: now[0])

    asyncio.run(limiter.wait("h1"))
    asyncio.run(limiter.wait("h1"))

    assert sleeps == [1.5]


def test_rate_limiter_is_per_host():
    sleeps = []

    async def _fake_sleep(seconds):
        sleeps.append(seconds)

    now = [100.0]
    limiter = _HostRateLimiter(1.0, sleep=_fake_sleep, clock=lambda: now[0])

    asyncio.run(limiter.wait("h1"))
    asyncio.run(limiter.wait("h2"))

    assert sleeps == [], "a different host must not inherit h1's wait"


def test_content_type_allowed_mapping():
    assert _content_type_allowed("text/html; charset=utf-8") == ".txt"
    assert _content_type_allowed("text/plain") == ".txt"
    assert _content_type_allowed("application/pdf") == ".pdf"
    assert _content_type_allowed("application/json") is None
    assert _content_type_allowed("image/png") is None


def test_parse_seeds_filters_unusable_entries():
    assert parse_seeds("http://a.com, ,ftp://b.com,https://c.com") == [
        "http://a.com",
        "https://c.com",
    ]
    assert parse_seeds("") == []


def test_crawl_without_seeds_raises_connector_error(fake_cfg):
    with pytest.raises(CrawlConnectorError):
        asyncio.run(crawl([], transport=_transport(lambda r: httpx.Response(200))))
