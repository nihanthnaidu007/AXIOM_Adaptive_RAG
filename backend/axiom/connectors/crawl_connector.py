"""AXIOM web-crawl connector (Wave 4, D4).

Polite BFS crawling over plain httpx fetches — no LLM calls, no metered
APIs, no headless browser (locked decision). Guardrails, all ON by default:

- robots.txt compliance via stdlib urllib.robotparser (absent robots.txt
  means unrestricted; disabling compliance is an explicit operator act).
- Per-host rate limit (politeness delay between fetches to the same host).
- Global page cap and depth cap; per-page byte cap (oversize fails visibly).
- Fetch-time dedup via a Redis set under ``axiom:crawl:`` — a namespace
  deliberately disjoint from ``axiom:cache:*`` so semantic-cache eviction
  can never touch crawl bookkeeping. Redis unavailable degrades to an
  in-process set for the run (logged, still deduped within the run).
- Connector-owned format allowlist on Content-Type: the HTTP endpoint's
  MIME sniffing applies to uploads, not crawl fetches (R11).

Everything funnels through the shared ingestion path via the same
ParseOutcome seam as uploads and the S3 connector; stable source keys are
the fetched URLs, making re-runs idempotent replacements.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Set
from urllib import robotparser
from urllib.parse import urljoin, urlsplit

import httpx

from axiom.config import get_config

logger = logging.getLogger(__name__)

CRAWL_DEDUP_NAMESPACE = "axiom:crawl:fetched"

# Connector-owned content-type allowlist (see module docstring re R11).
CRAWL_ALLOWED_CONTENT_TYPES = {"text/html", "text/plain", "application/pdf"}

_USER_AGENT = "AXIOM-Crawler/1.0"


class _HTMLTextExtractor(HTMLParser):
    """Stdlib HTML → text/link extraction (no extra dependency, no browser).

    Script/style content is dropped; href attributes are collected for BFS.
    """

    _SKIP_TAGS = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: List[str] = []
        self._skip_depth = 0
        self._in_title = False
        self.links: List[str] = []
        self.title = ""

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title" and not self.title:
            self._in_title = True
        for name, value in attrs:
            if name == "href" and value:
                self.links.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)
            if self._in_title and not self.title:
                self.title = text[:200]

    def text(self) -> str:
        return "\n".join(self._chunks)


class CrawlDedupStore:
    """Fetch-time dedup under axiom:crawl: with in-process fallback.

    Redis keeps dedup across runs; the fallback set (logged) still dedupes
    within one run when Redis is unavailable. Only successfully fetched and
    accepted pages are marked — a failed fetch is retried on the next run
    instead of being silently skipped forever.
    """

    def __init__(self, redis_client: Any = None) -> None:
        self._redis = redis_client
        self._local: Set[str] = set()

    async def seen(self, url: str) -> bool:
        """True if url was already fetched and accepted in a previous run."""
        if self._redis is not None:
            try:
                return bool(await self._redis.sismember(CRAWL_DEDUP_NAMESPACE, url))
            except Exception as exc:
                logger.warning(
                    "Crawl dedup Redis unavailable (%s) — falling back to in-process set",
                    exc,
                )
                self._redis = None
        return url in self._local

    async def mark_fetched(self, url: str) -> None:
        """Record a successful fetch so re-runs become idempotent replacements."""
        if self._redis is not None:
            try:
                await self._redis.sadd(CRAWL_DEDUP_NAMESPACE, url)
                return
            except Exception as exc:
                logger.warning(
                    "Crawl dedup Redis unavailable (%s) — falling back to in-process set",
                    exc,
                )
                self._redis = None
        self._local.add(url)


@dataclass
class CrawlFetchedPage:
    """One fetched page ready for the shared parse seam."""

    source: str  # stable source key: the URL
    filename: str
    content: bytes
    ext: str  # extension for the parse seam (.txt/.pdf)
    fetched_at: datetime
    content_hash: str
    size: int
    title: str


class CrawlConnectorError(RuntimeError):
    """Crawl-level failure (unusable seed config), distinct from per-page errors."""


def is_crawl_configured() -> bool:
    """True when at least one usable seed URL is configured. No network call."""
    return bool(parse_seeds(get_config().crawl_seeds))


def parse_seeds(raw: str) -> List[str]:
    """Split the CRAWL_SEEDS config into usable http(s) URLs."""
    seeds = [s.strip() for s in (raw or "").split(",") if s.strip()]
    return [s for s in seeds if urlsplit(s).scheme in ("http", "https")]


def _host_of(url: str) -> str:
    parts = urlsplit(url)
    return parts.netloc.lower()


class _HostRateLimiter:
    """Politeness delay per host (asyncio-friendly, injectable clock/sleep)."""

    def __init__(self, delay_seconds: float, sleep=asyncio.sleep, clock=time.monotonic):
        self._delay = delay_seconds
        self._sleep = sleep
        self._clock = clock
        self._last: Dict[str, float] = {}

    async def wait(self, host: str) -> None:
        now = self._clock()
        elapsed = now - self._last.get(host, 0.0)
        if elapsed < self._delay:
            await self._sleep(self._delay - elapsed)
        self._last[host] = self._clock()


class _RobotsCache:
    """Per-host robots.txt cache honoring crawl_respect_robots."""

    def __init__(self, client: httpx.AsyncClient, respect: bool):
        self._client = client
        self._respect = respect
        self._cache: Dict[str, Optional[robotparser.RobotFileParser]] = {}

    async def allowed(self, url: str) -> bool:
        if not self._respect:
            return True
        host = _host_of(url)
        if host not in self._cache:
            robots_url = urljoin(url, "/robots.txt")
            parser = robotparser.RobotFileParser()
            try:
                resp = await self._client.get(robots_url)
                if resp.status_code >= 400:
                    self._cache[host] = None  # absent/unauthorized robots: unrestricted
                else:
                    parser.parse(resp.text.splitlines())
                    self._cache[host] = parser
            except Exception:
                self._cache[host] = None  # unreachable robots: fail open for this host
        cached = self._cache[host]
        if cached is None:
            return True
        return cached.can_fetch(_USER_AGENT, url)


def _content_type_allowed(content_type: str) -> Optional[str]:
    """Map an allowed Content-Type to its parse-seam extension (None = skip)."""
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime not in CRAWL_ALLOWED_CONTENT_TYPES:
        return None
    if mime in ("text/html", "text/plain"):
        return ".txt"
    return ".pdf"


def _filename_for(url: str, ext: str, title: str) -> str:
    parts = urlsplit(url)
    tail = [p for p in parts.path.rsplit("/", 1)[-1].split(".") if p]
    name = " ".join(tail[:-1]) if len(tail) > 1 else (tail[0] if tail else "")
    name = name or title or parts.netloc.replace("www.", "")
    return f"{name[:80]}{ext}"


def _html_to_text(html: str) -> tuple[str, str]:
    """Extract text + title from HTML. Returns (text, title)."""
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html)
        extractor.close()
    except Exception as exc:
        # A malformed page should not kill the run: log and use whatever
        # extraction produced; the empty-content check reports no-shows.
        logger.warning("HTML extraction failed: %s", exc)
    return extractor.text(), extractor.title


def _extract_links(html: str, base_url: str) -> List[str]:
    """Resolve in-page hrefs against base_url; http(s) only, fragments dropped."""
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(html)
        extractor.close()
    except Exception:
        return []
    resolved: List[str] = []
    for href in extractor.links:
        absolute = urljoin(base_url, href)
        if urlsplit(absolute).scheme in ("http", "https"):
            parts = urlsplit(absolute)
            resolved.append(parts._replace(fragment="").geturl())
    return resolved


async def crawl(
    seed_urls: List[str],
    *,
    dedup: Optional[CrawlDedupStore] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    client_factory: Any = None,
) -> tuple[List[CrawlFetchedPage], List[Dict[str, str]]]:
    """BFS-crawl from seeds under the configured caps. Returns (pages, errors).

    Per-page failures land in ``errors`` with sanitized reasons; the crawl
    raises only when the seed configuration itself is unusable. Cross-host
    links are recorded but not followed — one run stays within the seeds'
    host (a scope-bomb guard; revisiting requires an operator setting).
    """
    cfg = get_config()
    if not seed_urls:
        raise CrawlConnectorError("No usable seed URLs configured for the crawl run.")

    dedup = dedup or CrawlDedupStore()
    rate = _HostRateLimiter(cfg.crawl_rate_limit_seconds)
    max_pages = max(cfg.crawl_max_pages, 0)
    max_depth = max(cfg.crawl_max_depth, 0)
    max_bytes = max(cfg.crawl_max_page_bytes, 0)

    pages: List[CrawlFetchedPage] = []
    errors: List[Dict[str, str]] = []

    headers = {"User-Agent": _USER_AGENT}
    if client_factory is not None:
        client = client_factory(headers=headers)
    else:
        client = httpx.AsyncClient(
            trust_env=True,  # honor proxy env, matching the web-search client
            follow_redirects=True,
            headers=headers,
            transport=transport,
        )
    robots = _RobotsCache(client, bool(cfg.crawl_respect_robots))

    try:
        queue: List[tuple[str, int]] = [(url, 0) for url in seed_urls]
        enqueued: Set[str] = set(seed_urls)

        while queue and len(pages) < max_pages:
            url, depth = queue.pop(0)
            if await dedup.seen(url):
                continue
            host = _host_of(url)
            if not await robots.allowed(url):
                errors.append({"source": url, "reason": "Blocked by robots.txt."})
                continue
            await rate.wait(host)

            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except Exception as exc:
                errors.append({
                    "source": url,
                    "reason": f"Fetch failed: {str(exc)[:200]}",
                })
                continue

            ext = _content_type_allowed(resp.headers.get("content-type", ""))
            if ext is None:
                errors.append({
                    "source": url,
                    "reason": (
                        "Content-Type not in the connector allowlist "
                        f"({resp.headers.get('content-type', 'unknown')})."
                    ),
                })
                continue
            if len(resp.content) > max_bytes:
                errors.append({
                    "source": url,
                    "reason": (
                        f"Response exceeds the byte cap "
                        f"({len(resp.content)} > {max_bytes})."
                    ),
                })
                continue

            if ext == ".txt" and "text/html" in resp.headers.get("content-type", ""):
                text, title = _html_to_text(resp.text)
                content = text.encode("utf-8")
            else:
                content = resp.content
                title = ""
            if not content.strip():
                errors.append({"source": url, "reason": "Page yielded no extractable content."})
                continue

            # Mark only after every acceptance gate passes: a failed or
            # rejected fetch stays unmarked so the next run retries it.
            await dedup.mark_fetched(url)
            pages.append(
                CrawlFetchedPage(
                    source=url,
                    filename=_filename_for(url, ext, title),
                    content=content,
                    ext=ext,
                    fetched_at=datetime.now(timezone.utc),
                    content_hash=hashlib.sha256(content).hexdigest(),
                    size=len(content),
                    title=title,
                )
            )

            if depth < max_depth and ext == ".txt":
                for href in _extract_links(resp.text, url):
                    if href not in enqueued and _host_of(href) == host:
                        enqueued.add(href)
                        queue.append((href, depth + 1))
    finally:
        if client_factory is None:
            await client.aclose()

    return pages, errors


def crawl_run_config_summary() -> Dict[str, Any]:
    """Non-secret run configuration for the run record / polling payload."""
    cfg = get_config()
    return {
        "seeds": parse_seeds(cfg.crawl_seeds),
        "max_pages": cfg.crawl_max_pages,
        "max_depth": cfg.crawl_max_depth,
        "rate_limit_seconds": cfg.crawl_rate_limit_seconds,
        "respect_robots": bool(cfg.crawl_respect_robots),
        "max_page_bytes": cfg.crawl_max_page_bytes,
    }
