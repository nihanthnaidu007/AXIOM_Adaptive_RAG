"""AXIOM S3 ingestion connector (Wave 4, D4).

Polls one S3 bucket (boto3, a normal pinned Apache-2.0 dependency) for
allowed-format documents and hands them to the shared ingestion path.

Design contracts locked by the W4 spec:

- ONE bucket per run (bucket + optional prefix, recursive or top-level).
- Extension allowlist owned by the connector: the HTTP endpoint's MIME
  sniffing applies to uploads, not S3 objects (R11).
- Per-object size cap: oversize objects fail visibly and are skipped.
- The RUN is the batch boundary: the caller indexes all fetched objects with
  one ``index_run`` call and clears the semantic cache once — never
  rebuild-per-add or cache-clear-per-doc.
- Stable source keys (``s3://bucket/key``) make re-runs idempotent: the
  replacement semantics of the indexing layer swap content per source.
- Missing S3 config is optional-disabled-with-warning (the Tavily pattern in
  axiom/search/web_search.py); configured-but-failing is per-object
  fail-visible (the indexer vector_error precedent).
"""

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

from axiom.config import get_config

logger = logging.getLogger(__name__)

# Connector-owned format allowlist (see module docstring re R11).
S3_ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".html", ".htm", ".csv"}

# Credential-shaped substrings must never reach a persisted error_reason.
_SECRET_PATTERNS = re.compile(
    r"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|"
    r"aws_secret_access_key\s*=\s*\S+|"
    r"aws_access_key_id\s*=\s*\S+|"
    r"Bearer\s+\S+|"
    r"arn:aws[a-z0-9-]*:[^\s\]]+|"  # IAM principals/topology
    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b)",  # internal IP-literal hosts
    re.IGNORECASE,
)


def sanitize_failure_reason(exc: BaseException | str) -> str:
    """Collapse an exception into a safe, bounded failure reason.

    Strips credential-shaped substrings, AWS URLs/hosts, IAM ARNs, and
    internal IP-literal hosts, caps length — the sanitized-reason contract
    for persisted run errors.
    """
    text = str(exc) if not isinstance(exc, str) else exc
    text = _SECRET_PATTERNS.sub("[redacted]", text)
    text = re.sub(r"https?://\S+", "[endpoint]", text)
    return text[:200]


def is_s3_configured() -> bool:
    """True when a bucket is configured. No network call (Tavily pattern)."""
    return bool(get_config().s3_bucket)


@dataclass
class S3FetchedObject:
    """One S3 object fetched and ready for the shared parse seam."""

    source: str  # stable source key: s3://bucket/key
    filename: str
    content: bytes
    fetched_at: datetime
    content_hash: str
    size: int


class S3ConnectorError(RuntimeError):
    """The bucket itself is unusable (auth, missing bucket, bad endpoint)."""


def _build_client(cfg: Any) -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - requirements pin boto3
        raise S3ConnectorError("boto3 is not installed") from exc
    params: Dict[str, Any] = {
        "region_name": cfg.s3_region,
        "aws_access_key_id": cfg.s3_access_key_id or None,
        "aws_secret_access_key": cfg.s3_secret_access_key or None,
    }
    if cfg.s3_endpoint_url:
        params["endpoint_url"] = cfg.s3_endpoint_url
    return boto3.client("s3", **params)


def _list_allowed_keys(client: Any, cfg: Any) -> List[Dict[str, Any]]:
    """List matching objects as {key, size} dicts (extension allowlist only).

    Non-recursive mode lists only top-level keys (Delimiter='/'), skipping
    common prefixes entirely. Size capping happens at fetch time in
    _collect_sync, against the same configured cap.
    """
    prefix = cfg.s3_prefix or ""
    listed: List[Dict[str, Any]] = []

    paginator = client.get_paginator("list_objects_v2")
    paginate_kwargs: Dict[str, Any] = {"Bucket": cfg.s3_bucket, "Prefix": prefix}
    if not cfg.s3_recursive:
        paginate_kwargs["Delimiter"] = "/"

    for page in paginator.paginate(**paginate_kwargs):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            size = int(obj.get("Size", 0))
            ext = _extension_of(key)
            if ext not in S3_ALLOWED_EXTENSIONS:
                continue
            if key.endswith("/"):
                continue
            listed.append({"key": key, "size": size})
    return listed


def _extension_of(key: str) -> str:
    from os.path import splitext

    return splitext(key)[1].lower()


def _collect_sync() -> tuple[List[S3FetchedObject], List[Dict[str, str]]]:
    """Synchronous collection core; runs in a worker thread via asyncio."""
    cfg = get_config()
    client = _build_client(cfg)
    max_bytes = max(cfg.s3_max_object_size_mb, 0) * 1024 * 1024

    try:
        candidates = _list_allowed_keys(client, cfg)
    except Exception as exc:
        # Bucket-level failure: fail the whole run with a sanitized reason.
        raise S3ConnectorError(
            f"S3 listing failed for bucket '{cfg.s3_bucket}': "
            f"{sanitize_failure_reason(exc)}"
        ) from exc

    fetched: List[S3FetchedObject] = []
    errors: List[Dict[str, str]] = []
    for item in candidates:
        key = item["key"]
        source = f"s3://{cfg.s3_bucket}/{key}"
        if item["size"] > max_bytes:
            errors.append({
                "source": source,
                "reason": (
                    f"Object exceeds the per-object cap "
                    f"({item['size']} bytes > {max_bytes})."
                ),
            })
            continue
        try:
            response = client.get_object(Bucket=cfg.s3_bucket, Key=key)
            body = response["Body"].read()
        except Exception as exc:
            # Per-object failure is visible in the run record, never silent.
            errors.append({
                "source": source,
                "reason": sanitize_failure_reason(exc),
            })
            continue
        fetched.append(
            S3FetchedObject(
                source=source,
                filename=key.rsplit("/", 1)[-1],
                content=body,
                fetched_at=datetime.now(timezone.utc),
                content_hash=hashlib.sha256(body).hexdigest(),
                size=len(body),
            )
        )
    return fetched, errors


async def fetch_s3_objects() -> tuple[List[S3FetchedObject], List[Dict[str, str]]]:
    """Fetch all allowed objects from the configured bucket.

    Runs boto3 (sync) in a worker thread. Returns fetched objects plus
    per-object errors with sanitized reasons. Raises S3ConnectorError when
    the bucket itself cannot be listed.
    """
    return await asyncio.to_thread(_collect_sync)


def s3_run_config_summary() -> Dict[str, Any]:
    """Non-secret run configuration for the run record / polling payload."""
    cfg = get_config()
    return {
        "bucket": cfg.s3_bucket,
        "prefix": cfg.s3_prefix or "",
        "recursive": bool(cfg.s3_recursive),
        "region": cfg.s3_region,
        "endpoint_configured": bool(cfg.s3_endpoint_url),
        "max_object_size_mb": cfg.s3_max_object_size_mb,
    }
