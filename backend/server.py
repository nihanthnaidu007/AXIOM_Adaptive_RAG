"""AXIOM FastAPI Backend - Main API Server."""

import asyncio
import hashlib
import inspect
import json
import logging
import os
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import magic
from dotenv import load_dotenv
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Security,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger(__name__)

QUERY_GRAPH_TIMEOUT_SEC = float(os.environ.get("QUERY_GRAPH_TIMEOUT_SEC", "180"))

from axiom.api_errors import (
    GENERIC_INTERNAL_MESSAGE,
    INTERNAL_ERROR,
    error_detail,
    sse_error_event,
)
from axiom.cache.semantic_cache import semantic_cache
from axiom.config import get_config
from axiom.connectors.crawl_connector import (
    CrawlDedupStore,
    crawl_run_config_summary,
    is_crawl_configured,
    parse_seeds,
)
from axiom.connectors.crawl_connector import (
    crawl as crawl_pages,
)
from axiom.connectors.s3_connector import (
    S3ConnectorError,
    fetch_s3_objects,
    is_s3_configured,
    s3_run_config_summary,
)
from axiom.db_migrations import upgrade_to_head
from axiom.evaluation.claude_evaluator import claude_evaluator
from axiom.graph.graph import get_graph, get_graph_node_names
from axiom.graph.state import create_initial_state
from axiom.graph.streaming import ContentSink, install_content_sink, reset_content_sink
from axiom.ingest.extract import ScannedPdfNotSupportedError, parse_document
from axiom.ingest.indexer import get_dual_indexer
from axiom.ingest.loader import DocumentChunker
from axiom.ingest.ocr import OcrExtraMissingError
from axiom.observability.langsmith import langsmith_tracer
from axiom.observability.logging import (
    configure_logging,
    get_request_id,
    normalize_request_id,
    request_id_var,
)
from axiom.observability.metrics import (
    begin_token_scope,
    end_token_scope,
    observe_query_tokens,
    observe_request,
    render_metrics,
)
from axiom.retrieval.bm25_index import bm25_index
from axiom.retrieval.reranker import get_reranker
from axiom.retrieval.vector_store import get_engine, vector_store

# Structured JSON logging (LOG_FORMAT=text for the legacy dev format).
# Installed after .env loads so LOG_FORMAT/LOG_LEVEL from the file apply.
configure_logging()

# System health state. Populated during lifespan startup.
# Feature 3 will extend this to all 5 components and expose it in every API response.
_system_health: dict = {
    "pgvector": "unknown",
    "redis": "unknown",
    "reranker": "unknown",
    "web_search": "unknown",
    "evaluator": "unknown",
    "generator": "unknown",
}

limiter = Limiter(key_func=get_remote_address)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    key: str = Security(_api_key_header),
) -> None:
    cfg = get_config()
    if not cfg.api_key:
        # Fail closed: without a configured key, authentication cannot succeed,
        # so protected endpoints refuse to serve instead of silently disabling auth.
        logger.error(
            "API_KEY is not configured — refusing request to a protected endpoint. "
            "Set API_KEY in the environment (or repo-root .env) and restart the server."
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "API authentication is not configured",
                "guidance": "Set API_KEY in the server environment (or repo-root .env) and restart. All protected endpoints refuse traffic until it is set.",
            },
        )
    if not key or key != cfg.api_key:
        raise HTTPException(status_code=401, detail={"error": "Invalid API key"})


@asynccontextmanager
async def lifespan(app):
    if not get_config().api_key:
        logger.warning(
            "API_KEY is not set — every protected endpoint will return 503 "
            "until it is configured. This is intentional fail-closed behavior."
        )

    checkpointer_stack = AsyncExitStack()
    await checkpointer_stack.__aenter__()
    app.state._checkpointer_stack = checkpointer_stack

    if get_config().run_migrations_on_startup:
        try:
            # Alembic owns the schema; env.py needs a loop-free thread.
            await asyncio.to_thread(upgrade_to_head)
            logger.info("Database migrations applied (alembic upgrade head)")
        except Exception as exc:
            # Consistent with the rest of startup: degrade, don't kill boot —
            # an unmigrated schema degrades features, a crashed boot serves nothing.
            logger.warning("Startup migration failed — continuing: %s", exc)
    else:
        logger.info("RUN_MIGRATIONS_ON_STARTUP=false — skipping startup migrations")

    connected = await vector_store.connect()
    if connected:
        logger.info("pgvector connected — chunk_embeddings table ready")
        await _hydrate_bm25_from_pgvector()
        await _hydrate_ingested_docs()
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        pg_saver = await checkpointer_stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(get_config().postgres_url)
        )
        await pg_saver.setup()
        app.state.checkpointer = pg_saver
    else:
        logger.warning("pgvector connection failed — vector retrieval will use fallback")
        from langgraph.checkpoint.memory import MemorySaver

        app.state.checkpointer = MemorySaver()
        logger.warning("PostgreSQL unavailable — using MemorySaver fallback")
    _system_health["pgvector"] = "connected" if connected else "not_connected"

    cache_connected = await semantic_cache.connect()
    if cache_connected:
        logger.info("Redis semantic cache connected")
    else:
        logger.warning("Redis cache connection failed — cache disabled")
    _system_health["redis"] = "connected" if cache_connected else "not_connected"

    cfg_eval = get_config()
    if cfg_eval.use_claude_evaluator:
        evaluator_available = await claude_evaluator.is_available()
        if evaluator_available:
            logger.info(
                "Claude evaluator ready — real RAGAS evaluation enabled "
                "(model: claude-haiku-4-5-20251001)"
            )
            _system_health["evaluator"] = "claude-haiku"
        else:
            logger.warning(
                "Claude evaluator ping failed at startup. "
                "Check ANTHROPIC_API_KEY and network access to api.anthropic.com. "
                "Evaluation calls will retry independently on each query."
            )
            _system_health["evaluator"] = "claude-haiku/unreachable"
    else:
        from axiom.evaluation.critic_llm import critic_llm

        ollama_connected = await critic_llm.connect()
        if ollama_connected:
            logger.info("Ollama critic connected — real RAGAS evaluation enabled")
            _system_health["evaluator"] = "ollama"
        else:
            logger.warning(
                "Ollama not available and USE_CLAUDE_EVALUATOR=false. "
                "Evaluation will produce parse_error results. "
                "Set USE_CLAUDE_EVALUATOR=true to use cloud evaluation."
            )
            _system_health["evaluator"] = "ollama/unavailable"

    # Generator probe (W3): mirrors the evaluator probe — the configured
    # generation backend is checked once at boot so a downed or unpulled
    # model is visible in /api/health and in stub mode instead of only
    # surfacing as per-query failures.
    cfg_gen = get_config()
    if cfg_gen.llm_provider == "cloud":
        _system_health["generator"] = "anthropic"
    else:
        from axiom.llm.client import llm_client

        generator_up = await llm_client.probe()
        if generator_up:
            logger.info("Ollama generator ready (model: %s)", cfg_gen.ollama_generation_model)
            _system_health["generator"] = f"ollama/{cfg_gen.ollama_generation_model}"
        else:
            logger.warning(
                "Ollama generator probe failed at startup (model: %s, host: %s). "
                "Query generation will fail visibly until the model is available — "
                "pull it (`ollama pull %s`) or switch LLM_PROVIDER back to cloud.",
                cfg_gen.ollama_generation_model,
                cfg_gen.ollama_host,
                cfg_gen.ollama_generation_model,
            )
            _system_health["generator"] = "ollama/unavailable"

    get_reranker().load()
    _system_health["reranker"] = "loaded" if get_reranker().is_loaded() else "not_loaded"
    logger.info("Reranker: %s", _system_health["reranker"])

    from axiom.search.web_search import is_tavily_configured

    _system_health["web_search"] = "tavily" if is_tavily_configured() else "not_configured"
    logger.info("Web search: %s", _system_health["web_search"])

    logger.info(
        "System health at startup: pgvector=%s redis=%s reranker=%s "
        "web_search=%s evaluator=%s generator=%s",
        _system_health["pgvector"],
        _system_health["redis"],
        _system_health["reranker"],
        _system_health["web_search"],
        _system_health["evaluator"],
        _system_health["generator"],
    )

    yield

    # --- Graceful shutdown cleanup ---
    logger.info("Shutdown: closing checkpointer connection...")
    try:
        await checkpointer_stack.aclose()
        logger.info("Shutdown: checkpointer closed")
    except Exception as exc:
        logger.warning("Shutdown: checkpointer cleanup error: %s", exc)

    logger.info("Shutdown: closing PostgreSQL connection pool…")
    try:
        if vector_store._engine:
            await vector_store._engine.dispose()
            logger.info("Shutdown: PostgreSQL connection pool closed")
    except Exception as exc:
        logger.warning("Shutdown: PostgreSQL cleanup error: %s", exc)

    logger.info("Shutdown: closing Redis connection…")
    try:
        if semantic_cache._redis:
            await semantic_cache._redis.aclose()
            logger.info("Shutdown: Redis connection closed")
    except Exception as exc:
        logger.warning("Shutdown: Redis cleanup error: %s", exc)

    if not get_config().use_claude_evaluator:
        from axiom.evaluation.critic_llm import critic_llm

        logger.info("Shutdown: closing Ollama httpx client...")
        try:
            if critic_llm._client:
                await critic_llm._client.aclose()
                logger.info("Shutdown: Ollama httpx client closed")
        except Exception as exc:
            logger.warning("Shutdown: Ollama httpx cleanup error: %s", exc)

    # Close Anthropic httpx client (claude_evaluator)
    if claude_evaluator._client is not None:
        try:
            await claude_evaluator._client.close()
            logger.info("Shutdown: Claude evaluator httpx client closed")
        except Exception as exc:
            logger.warning("Shutdown: Claude evaluator cleanup error: %s", exc)

    # Close generation LLM client (cloud Anthropic or local Ollama transport)
    from axiom.llm.client import llm_client

    try:
        await llm_client.aclose()
        logger.info("Shutdown: Generation LLM client closed")
    except Exception as exc:
        logger.warning("Shutdown: Generation LLM cleanup error: %s", exc)

    logger.info("Shutdown: cleanup complete")


async def _hydrate_ingested_docs():
    """Populate the in-memory _ingested_docs list from PostgreSQL on startup."""
    try:
        from sqlalchemy import text as sa_text

        async with get_engine().connect() as conn:
            rows = await conn.execute(
                sa_text(
                    "SELECT doc_id, filename, chunk_count, file_size_bytes, indexed_at "
                    "FROM ingested_documents ORDER BY indexed_at"
                )
            )
            for r in rows:
                row = dict(r._mapping)
                _ingested_docs.append(
                    {
                        "doc_id": row["doc_id"],
                        "filename": row["filename"],
                        "chunk_count": row["chunk_count"],
                        "indexed_at": row["indexed_at"].isoformat()
                        if hasattr(row["indexed_at"], "isoformat")
                        else str(row["indexed_at"]),
                        "status": "indexed",
                    }
                )
        if _ingested_docs:
            logger.info(
                "Hydrated %d ingested document records from PostgreSQL", len(_ingested_docs)
            )
    except Exception as exc:
        logger.warning("Failed to hydrate ingested docs: %s", exc)


async def _persist_trace(session_id: str, trace_data: list) -> None:
    """Upsert a trace to PostgreSQL (write-through)."""
    if not get_engine():
        return
    try:
        from sqlalchemy import text as sa_text

        async with get_engine().begin() as conn:
            await conn.execute(
                sa_text("""
                INSERT INTO pipeline_traces (session_id, trace_data)
                VALUES (:sid, :data)
                ON CONFLICT (session_id) DO UPDATE SET trace_data = :data, created_at = NOW()
            """),
                {"sid": session_id, "data": json.dumps(trace_data)},
            )
    except Exception as exc:
        logger.warning("Failed to persist trace %s: %s", session_id, exc)


async def _load_trace(session_id: str) -> list | None:
    """Load a trace from PostgreSQL. Returns None if not found."""
    if not get_engine():
        return None
    try:
        from sqlalchemy import text as sa_text

        async with get_engine().connect() as conn:
            row = await conn.execute(
                sa_text("SELECT trace_data FROM pipeline_traces WHERE session_id = :sid"),
                {"sid": session_id},
            )
            result = row.fetchone()
            if result:
                # asyncpg hands JSONB back as a raw string — decode to the
                # list the /trace contract promises.
                data = result[0]
                if isinstance(data, str):
                    data = json.loads(data)
                return data
        return None
    except Exception as exc:
        logger.warning("Failed to load trace %s: %s", session_id, exc)
        return None


async def _invoke_graph_with_metrics(
    graph: Any,
    initial_state: Any,
    config: Dict[str, Any],
    endpoint: str,
) -> Dict[str, Any]:
    """Run one graph invocation with per-request token accounting.

    Usage is reported by the LLM/embedding clients into the active token
    scope; closing the scope here attributes the totals to this endpoint.
    """
    scope = begin_token_scope()
    try:
        return await asyncio.wait_for(
            graph.ainvoke(initial_state, config=config),
            timeout=QUERY_GRAPH_TIMEOUT_SEC,
        )
    finally:
        prompt_tokens, completion_tokens = end_token_scope(scope)
        if prompt_tokens or completion_tokens:
            observe_query_tokens(endpoint, prompt_tokens, completion_tokens)


def _serialize_model(obj: Any) -> Any:
    """Best-effort JSON-safe serialization for graph state values."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return str(obj)


def _sources_from_state(final_state: dict) -> dict:
    """Build the SSE ``sources`` (citations) payload from final graph state.

    Document chunks come from the reranked corpus; web results from the
    web-fallback node. Capped and de-duplicated so a large corpus cannot
    balloon the terminal frames.
    """
    sources: list[dict] = []
    seen: set[tuple[object, ...]] = set()

    for c in (final_state.get("reranked_chunks") or [])[:10]:
        chunk = _serialize_model(c) or {}
        if not isinstance(chunk, dict):
            continue
        doc_key = ("document", chunk.get("source"), chunk.get("chunk_id"))
        if doc_key in seen:
            continue
        seen.add(doc_key)
        sources.append(
            {
                "kind": "document",
                "chunk_id": chunk.get("chunk_id"),
                "source": chunk.get("source"),
                "score": chunk.get("rerank_score")
                if chunk.get("rerank_score") is not None
                else chunk.get("rrf_score"),
                "preview": (chunk.get("content") or "")[:120],
                # Page provenance (Wave 4): None on pre-W4 rows — the SSE
                # passthrough stays additive, no frontend contract break.
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "origin_type": chunk.get("origin_type"),
            }
        )

    for w in (final_state.get("web_search_chunks") or [])[:5]:
        if not isinstance(w, dict):
            continue
        web_key = ("web", w.get("url"))
        if web_key in seen:
            continue
        seen.add(web_key)
        sources.append(
            {
                "kind": "web",
                "url": w.get("url"),
                "title": w.get("title"),
                "score": w.get("score"),
            }
        )

    return {
        "type": "sources",
        "sources": sources,
        "web_search_used": final_state.get("web_search_used", False),
    }


CITATION_HOST_NODES = ("rerank_chunks", "check_cache")
MAX_PERSISTED_CITATIONS = 10
CITATION_CONTENT_EXCERPT_CHARS = 600


def _citations_from_state(final_state: dict) -> list[dict]:
    """Chunk-level citation records from the final graph state.

    Same source the live payload uses (``reranked_chunks``), projected to the
    citation fields the panel and the historical endpoint serve. Content is
    excerpted and the list capped so a large corpus cannot balloon trace_data.
    """
    citations: list[dict] = []
    for c in (final_state.get("reranked_chunks") or [])[:MAX_PERSISTED_CITATIONS]:
        chunk = _serialize_model(c) or {}
        if not isinstance(chunk, dict) or not chunk.get("chunk_id"):
            continue
        citations.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "source": chunk.get("source"),
                "content": (chunk.get("content") or "")[:CITATION_CONTENT_EXCERPT_CHARS],
                "bm25_score": chunk.get("bm25_score"),
                "vector_score": chunk.get("vector_score"),
                "rrf_score": chunk.get("rrf_score"),
                "rerank_score": chunk.get("rerank_score"),
                "pre_rerank_position": chunk.get("pre_rerank_position"),
                "post_rerank_position": chunk.get("post_rerank_position"),
                # Page provenance (Wave 4): page span + origin ride citations
                # into the persisted trace so history replay keeps provenance.
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "origin_type": chunk.get("origin_type"),
            }
        )
    return citations


def _attach_trace_citations(trace_steps: list, citations: list) -> None:
    """Persist chunk-level citations into the trace's JSONB (history replay).

    Enriches the rerank step's ``detail`` in place — trace_data stays a list
    of trace steps, so /trace consumers are unaffected. Cache-hit runs have
    no rerank step; their chunks ride the check_cache step instead. A run
    with neither (or no citations) persists nothing.
    """
    if not citations:
        return
    for node_name in CITATION_HOST_NODES:
        for step in reversed(trace_steps):
            if not isinstance(step, dict) or step.get("node_name") != node_name:
                continue
            detail = step.get("detail")
            if not isinstance(detail, dict):
                detail = {}
                step["detail"] = detail
            detail["citations"] = citations
            return


def _extract_trace_citations(trace_steps: list) -> list[dict]:
    """Read the persisted citations back out of a trace_data list."""
    for step in reversed(trace_steps):
        if not isinstance(step, dict):
            continue
        detail = step.get("detail")
        if isinstance(detail, dict) and detail.get("citations"):
            return detail["citations"]
    return []


def _make_doc_id(filename: str) -> str:
    """Generate a unique lineage id for one ingest of a document."""
    return hashlib.sha256(
        f"{filename}:{datetime.now(timezone.utc).isoformat()}".encode()
    ).hexdigest()[:16]


async def _persist_ingested_doc(
    doc_id: str,
    filename: str,
    chunk_count: int,
    file_size_bytes: int,
    *,
    origin_type: str = "upload",
    origin_uri: str | None = None,
    fetched_at: datetime | None = None,
    content_hash: str | None = None,
    status: str = "completed",
    error_reason: str | None = None,
    parse_confidence: float | None = None,
) -> None:
    """Insert an ingested document record to PostgreSQL with W4 provenance."""
    if not get_engine():
        return
    try:
        from sqlalchemy import text as sa_text

        async with get_engine().begin() as conn:
            await conn.execute(
                sa_text("""
                INSERT INTO ingested_documents
                    (doc_id, filename, chunk_count, file_size_bytes,
                     origin_type, origin_uri, fetched_at, content_hash,
                     status, error_reason, parse_confidence)
                VALUES (:did, :fn, :cc, :fsb,
                        :otype, :ouri, :fetched, :chash,
                        :status, :reason, :pconf)
                ON CONFLICT (doc_id) DO NOTHING
            """),
                {
                    "did": doc_id,
                    "fn": filename,
                    "cc": chunk_count,
                    "fsb": file_size_bytes,
                    "otype": origin_type,
                    "ouri": origin_uri,
                    "fetched": fetched_at,
                    "chash": content_hash,
                    "status": status,
                    "reason": error_reason,
                    "pconf": parse_confidence,
                },
            )
    except Exception as exc:
        logger.warning("Failed to persist ingested doc %s: %s", filename, exc)


async def _delete_ingested_doc_record(doc_id: str) -> None:
    """Remove a document lineage row from PostgreSQL."""
    if not get_engine():
        return
    from sqlalchemy import text as sa_text

    async with get_engine().begin() as conn:
        await conn.execute(
            sa_text("DELETE FROM ingested_documents WHERE doc_id = :did"),
            {"did": doc_id},
        )


def _to_pg_timestamp(value: Any) -> Any:
    """ISO string -> datetime for asyncpg's timestamptz encoder (None on junk)."""
    if value is None or isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _json_or_none(value: Any) -> Any:
    """Decode asyncpg's raw-string JSONB (or pass any JSON value through)."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


async def _upsert_eval_run(job_id: str, job_data: Dict[str, Any]) -> None:
    """Mirror the full job status row into PostgreSQL.

    Written at start, on every progress tick, and at completion so any
    worker can serve GET /eval/status — the in-process dict alone is what
    used to pin eval jobs to a single worker.
    """
    if not get_engine():
        return
    try:
        from sqlalchemy import text as sa_text

        async with get_engine().begin() as conn:
            await conn.execute(
                sa_text("""
                INSERT INTO eval_runs (job_id, status, progress, total, aggregate, results, error, latest, started_at, completed_at)
                VALUES (:jid, :st, :pr, :tot, :agg, :res, :err, :lat, :sa, :ca)
                ON CONFLICT (job_id) DO UPDATE SET
                    status = :st, progress = :pr, total = :tot, aggregate = :agg, results = :res,
                    error = :err, latest = :lat, completed_at = :ca
            """),
                {
                    "jid": job_id,
                    "st": job_data.get("status"),
                    "pr": job_data.get("progress", 0),
                    "tot": job_data.get("total", 0),
                    "agg": json.dumps(job_data.get("aggregate")),
                    "res": json.dumps(job_data.get("results")),
                    "err": job_data.get("error"),
                    "lat": json.dumps(job_data.get("latest")),
                    "sa": _to_pg_timestamp(job_data.get("started_at")),
                    "ca": _to_pg_timestamp(job_data.get("completed_at")),
                },
            )
    except Exception as exc:
        logger.warning("Failed to persist eval run %s: %s", job_id, exc)


async def _load_eval_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Eval job status: PostgreSQL first (any worker), in-memory fallback."""
    engine = get_engine()
    if engine is not None:
        try:
            from sqlalchemy import text as sa_text

            async with engine.connect() as conn:
                row = (
                    await conn.execute(
                        sa_text(
                            "SELECT status, progress, total, aggregate, results, error, latest, "
                            "started_at, completed_at FROM eval_runs WHERE job_id = :jid"
                        ),
                        {"jid": job_id},
                    )
                ).fetchone()
            if row:
                job = row._mapping
                started_at = job["started_at"]
                completed_at = job["completed_at"]
                return {
                    "status": job["status"],
                    "progress": job["progress"] or 0,
                    "total": job["total"] or 0,
                    "results": _json_or_none(job["results"]) or [],
                    "latest": _json_or_none(job["latest"]),
                    "aggregate": _json_or_none(job["aggregate"]),
                    "error": job["error"],
                    "started_at": started_at.isoformat() if started_at else None,
                    "completed_at": completed_at.isoformat() if completed_at else None,
                }
        except Exception as exc:
            logger.warning("Failed to load eval job %s from PostgreSQL: %s", job_id, exc)
    job = _eval_jobs.get(job_id)
    return dict(job) if job else None


async def _hydrate_bm25_from_pgvector():
    """Load chunks from pgvector into in-memory BM25 so both indexes stay in sync."""
    try:
        from sqlalchemy import text as sa_text

        async with get_engine().connect() as conn:
            rows = await conn.execute(
                sa_text(
                    "SELECT chunk_id, source, content, chunk_index, token_count FROM chunk_embeddings"
                )
            )
            chunks = [dict(r._mapping) for r in rows]
        if chunks:
            await bm25_index.add_chunks(chunks)
            logger.info("BM25 hydrated from pgvector — %d chunks loaded", len(chunks))
    except Exception as exc:
        logger.warning("BM25 hydration from pgvector failed: %s", exc)


app = FastAPI(
    title="AXIOM Intelligence Platform",
    description="Adaptive RAG Intelligence System with Self-Correcting Hallucination Detection",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
# slowapi's handler takes RateLimitExceeded, narrower than starlette's
# Exception protocol — the runtime contract is fine, the typing isn't.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api")


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    # /api/ingest handles its own size check (50 MB) — skip the query-body limit for it
    if request.url.path.rstrip("/") == "/api/ingest":
        return await call_next(request)
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 50 * 1024:
        return JSONResponse(
            status_code=413,
            content={"error": "Request body too large — maximum 50KB"},
        )
    return await call_next(request)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    return response


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Assign/propagate the per-request ID. Registered LAST so it is the
    outermost middleware: Starlette's add_middleware inserts at index 0, so
    the last-registered layer wraps all others. Every response — including
    early returns from inner middleware such as the 413 body-limit
    rejection — carries X-Request-ID, gets a completion log line with that
    ID, and feeds the metrics counters.

    The ID comes from the client's X-Request-ID header when it is safe to
    echo, else a fresh uuid4 — so gateway-correlation and log greps agree.
    """
    request_id = normalize_request_id(request.headers.get("x-request-id", "")) or uuid.uuid4().hex
    request.state.request_id = request_id
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception:
            # Last-resort envelope for anything no route handled. Logged with
            # the request ID still bound; the client gets the same sanitized
            # shape as route-level 500s.
            logger.exception(
                "Unhandled exception (request_id=%s, path=%s)", request_id, request.url.path
            )
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": error_detail(
                        INTERNAL_ERROR,
                        GENERIC_INTERNAL_MESSAGE,
                        request_id=request_id,
                    )
                },
            )
        response.headers["X-Request-ID"] = request_id
        # Route template (e.g. /api/query) keeps the endpoint label cardinality
        # bounded; unmatched paths (404s) collapse into one bucket.
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        observe_request(request.method, endpoint, response.status_code, time.perf_counter() - start)
        logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "http_method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            },
        )
        return response
    finally:
        request_id_var.reset(token)


@app.exception_handler(StarletteHTTPException)
async def http_exception_with_request_id(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Inject the request ID into every HTTPException envelope's context.

    Additive only: detail.error stays a plain string for existing consumers,
    and context keys route code already supplied are preserved.
    """
    detail: Any = exc.detail
    if isinstance(detail, dict) and isinstance(detail.get("context"), dict):
        # Enrich only envelopes that already carry a context object — the
        # W1 sanitized-envelope shapes (e.g. 401 {"error": ...}) stay
        # byte-for-byte intact. The ID is always on the X-Request-ID
        # response header regardless.
        detail = dict(detail)
        request_id = get_request_id()
        if request_id:
            context = dict(detail["context"])
            context.setdefault("request_id", request_id)
            detail["context"] = context
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": detail},
        headers=exc.headers,
    )


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Prometheus scrape endpoint — public like /health (exposes no query data)."""
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


# --- Pydantic Models ---


class QueryRequest(BaseModel):
    query: str = Field(..., description="The natural language query to process")
    session_id: Optional[str] = Field(default=None, description="Session ID for trace grouping")


class QueryResponse(BaseModel):
    session_id: str
    final_answer: str
    confidence: Optional[Dict[str, Any]]
    classification: Optional[Dict[str, Any]]
    retrieval_strategy: str
    ragas_scores: Optional[Dict[str, Any]]
    scores_history: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    correction_attempts: int
    correction_history: List[Dict[str, Any]]
    trace_steps: List[Dict[str, Any]]
    served_from_cache: bool
    is_complete: bool
    error: Optional[str]
    total_latency_ms: Optional[float] = None
    parallel_timing: Optional[Dict[str, Any]] = None
    cache_result: Optional[Dict[str, Any]] = None
    langsmith_trace_url: Optional[str] = None
    decomposed: bool = False
    sub_query_results: List[Dict[str, Any]] = []
    evaluation_mode: str = "unknown"
    web_search_used: bool = False
    web_search_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    document_chunk_count: int = 0
    web_chunk_count: int = 0
    system_health: Dict[str, str] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    filename: str
    chunk_count: int
    status: str
    doc_id: Optional[str] = None
    mode: Optional[str] = None
    bm25: Optional[str] = None
    vector: Optional[str] = None
    chunks: Optional[List[Dict[str, Any]]] = None


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    filename: str
    deleted_chunks: int
    cache_keys_cleared: int
    status: str


class ConnectorRunResponse(BaseModel):
    """Accepted background connector run (poll it at /connectors/runs/{id})."""

    run_id: str
    connector: str
    status: str


class ConnectorRunDocument(BaseModel):
    """Per-document outcome inside a connector run."""

    source: str
    origin_type: str
    origin_uri: str
    fetched_at: Optional[str] = None
    content_hash: Optional[str] = None
    size: Optional[int] = None
    chunk_count: int = 0
    status: str = "completed"
    parse_confidence: Optional[float] = None
    error: Optional[str] = None


class ConnectorRunStatusResponse(BaseModel):
    """Full run record for the authenticated polling endpoint."""

    run_id: str
    connector: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    documents: List[ConnectorRunDocument] = Field(default_factory=list)
    errors: List[Dict[str, str]] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    indexed_chunks: Optional[int] = None
    cache_cleared: bool = False


class TraceResponse(BaseModel):
    session_id: str
    trace_steps: List[Dict[str, Any]]


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., description="Session/trace id the feedback applies to")
    rating: Literal[1, -1] = Field(..., description="+1 thumbs-up, -1 thumbs-down")
    comment: Optional[str] = Field(
        default=None, max_length=2000, description="Optional free-text context"
    )
    query_snippet: Optional[str] = Field(
        default=None, max_length=200, description="Query text at submit time"
    )


# --- In-memory stores ---
_trace_store: Dict[str, List[Dict[str, Any]]] = {}
_ingested_docs: List[Dict[str, Any]] = []
# Eval jobs: job_id -> status payload (background suite + polling)
_eval_jobs: Dict[str, Dict[str, Any]] = {}

_eval_semaphore = asyncio.Semaphore(1)
_ingest_semaphore = asyncio.Semaphore(3)


# --- API Endpoints ---


@api_router.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "AXIOM Intelligence Platform v1.0", "status": "operational"}


def _compute_stub_mode() -> bool:
    """Derive stub_mode from startup health state. No network calls.

    Stub mode is True when any critical pipeline component is unavailable:
    - pgvector not connected (retrieval fails entirely)
    - evaluator unreachable (cannot produce trustworthy scores)
    - reranker not loaded (ranking falls back to raw scores)
    - generator unknown/unavailable (queries would fail visibly — W3)

    Redis and web_search are not included: Redis down degrades cache
    performance but does not block the pipeline. Web search absent is
    normal in document-only deployments.
    """
    pgvector_ok = _system_health.get("pgvector") == "connected"
    evaluator_str = _system_health.get("evaluator", "unknown")
    evaluator_ok = (
        "unavailable" not in evaluator_str
        and "unreachable" not in evaluator_str
        and evaluator_str != "unknown"
    )
    reranker_ok = _system_health.get("reranker") == "loaded"
    generator_str = _system_health.get("generator", "unknown")
    generator_ok = "unavailable" not in generator_str and generator_str != "unknown"
    return not pgvector_ok or not evaluator_ok or not reranker_ok or not generator_ok


@api_router.get("/health")
async def health_check():
    try:
        graph = get_graph()
        graph_compiled = graph is not None
    except Exception:
        graph_compiled = False

    reranker = get_reranker()
    pg_connected = await vector_store.is_connected()
    vec_count = await vector_store.count() if pg_connected else 0

    return {
        "status": "ok",
        "graph_compiled": graph_compiled,
        "nodes": get_graph_node_names(),
        "stub_mode": _compute_stub_mode(),
        "system_health": dict(_system_health),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "index_status": {
            "bm25": "ready",
            "bm25_doc_count": bm25_index.count(),
            "vector": "ready" if pg_connected else "not_connected",
            "vector_doc_count": vec_count,
            "reranker": "loaded" if reranker.is_loaded() else "not_loaded",
        },
        "services": {
            "postgres": _system_health.get("pgvector", "unknown"),
            "redis": _system_health.get("redis", "unknown"),
            "evaluator": _system_health.get("evaluator", "unknown"),
            "generator": _system_health.get("generator", "unknown"),
            "web_search": _system_health.get("web_search", "unknown"),
            "reranker": _system_health.get("reranker", "unknown"),
        },
        "langsmith": "enabled"
        if langsmith_tracer.is_enabled()
        else "disabled (set LANGCHAIN_TRACING_V2=true)",
        "checkpointing": "enabled (MemorySaver)",
    }


@api_router.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def process_query(request: Request, body: QueryRequest):
    _start_time = time.time()

    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail={"error": "Query cannot be empty"})
    if len(body.query) > get_config().max_query_length:
        raise HTTPException(
            status_code=400, detail={"error": "Query too long — maximum 2000 characters"}
        )

    session_id = body.session_id
    if session_id:
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(
                status_code=400, detail={"error": "session_id must be a valid UUID"}
            )
    else:
        session_id = str(uuid.uuid4())
    current_node = None

    try:
        initial_state = create_initial_state(user_query=body.query, session_id=session_id)

        graph = get_graph(checkpointer=app.state.checkpointer)

        langsmith_config = langsmith_tracer.get_run_config(
            run_name=f"axiom-query-{session_id[:8]}",
            session_id=session_id,
            metadata={
                "user_query": body.query[:100],
                "stub_mode": _compute_stub_mode(),
            },
        )

        full_config = {
            **langsmith_config,
            "configurable": {"thread_id": session_id},
        }

        try:
            final_state = await _invoke_graph_with_metrics(
                graph, initial_state, full_config, endpoint="/api/query"
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail={
                    "error": f"Query timed out after {QUERY_GRAPH_TIMEOUT_SEC:.0f}s",
                    "session_id": session_id,
                },
            )

        final_state["langsmith_trace_url"] = langsmith_tracer.get_trace_url(session_id)

        trace_steps = final_state.get("trace_steps", [])
        _trace_store[session_id] = [
            step.model_dump() if hasattr(step, "model_dump") else dict(step) for step in trace_steps
        ]
        _attach_trace_citations(_trace_store[session_id], _citations_from_state(final_state))
        await _persist_trace(session_id, _trace_store[session_id])

        def serialize_model(obj):
            if obj is None:
                return None
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if isinstance(obj, dict):
                return obj
            return str(obj)

        parallel_timing = final_state.get("parallel_timing")

        ragas = final_state.get("ragas_scores")
        eval_mode = (
            ragas.evaluation_mode if ragas and hasattr(ragas, "evaluation_mode") else "unknown"
        )

        return QueryResponse(
            session_id=session_id,
            final_answer=final_state.get("final_answer", ""),
            confidence=serialize_model(final_state.get("confidence")),
            classification=serialize_model(final_state.get("classification")),
            retrieval_strategy=final_state.get("retrieval_strategy", ""),
            ragas_scores=serialize_model(ragas),
            scores_history=[serialize_model(s) for s in final_state.get("scores_history", [])],
            reranked_chunks=[serialize_model(c) for c in final_state.get("reranked_chunks", [])],
            correction_attempts=final_state.get("correction_attempts", 0),
            correction_history=[
                serialize_model(c) for c in final_state.get("correction_history", [])
            ],
            trace_steps=[serialize_model(s) for s in trace_steps],
            served_from_cache=final_state.get("served_from_cache", False),
            is_complete=final_state.get("is_complete", False),
            error=final_state.get("error"),
            total_latency_ms=round((time.time() - _start_time) * 1000, 2),
            parallel_timing=serialize_model(parallel_timing),
            cache_result=serialize_model(final_state.get("cache_result")),
            langsmith_trace_url=final_state.get("langsmith_trace_url"),
            decomposed=final_state.get("decomposed", False),
            sub_query_results=final_state.get("sub_query_results", []),
            evaluation_mode=eval_mode,
            web_search_used=final_state.get("web_search_used", False),
            web_search_chunks=final_state.get("web_search_chunks", []),
            document_chunk_count=final_state.get("document_chunk_count", 0),
            web_chunk_count=final_state.get("web_chunk_count", 0),
            system_health=dict(_system_health),
        )

    except HTTPException:
        # Intentional envelopes (timeout 504, validation 400) pass through
        # untouched — the generic handler below must not re-wrap them.
        raise
    except Exception:
        logger.exception("Query failed (node=%s, session=%s)", current_node, session_id)
        error_trace = [
            {
                "node_name": current_node or "unknown",
                "status": "error",
                "summary": "Query processing failed",
                "detail": {"code": INTERNAL_ERROR},
            }
        ]
        _trace_store[session_id] = error_trace
        await _persist_trace(session_id, error_trace)

        raise HTTPException(
            status_code=500,
            detail=error_detail(
                INTERNAL_ERROR,
                GENERIC_INTERNAL_MESSAGE,
                session_id=session_id,
                node=current_node,
            ),
        )


@api_router.post("/query/stream", dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def query_stream(request: Request, body: QueryRequest):
    """SSE streaming endpoint — progressive answer generation.

    Frame sequence (every ``data:`` payload is JSON with a ``type`` field and
    the request's ``request_id`` for log correlation):

    - ``status``         — pipeline stage transitions (retrieving/generating)
    - ``node_complete``  — one per graph node, as it completes (unchanged shape)
    - ``content``        — chunk-level answer text deltas, in generation order
                           (also emitted for semantic-cache hits)
    - ``sources``        — final citations (reranked document chunks + web
                           results), immediately before the terminal event
    - ``done``           — the full QueryResponse payload (same shape as the
                           /query JSON body)
    - ``error``          — sanitized failure envelope; stream then ends
    - ``data: [DONE]``   — stream sentinel, always the last frame

    Backward compatibility: the JSON ``/api/query`` response is untouched and
    the ``done`` event keeps its established shape — new event types are
    additive, and clients that only understand ``node_complete``/``done``/
    ``error`` keep working.

    Disconnect semantics: a client that goes away mid-stream has its graph
    producer task cancelled — LangGraph propagates cancellation into the
    running node, the Anthropic stream context closes, and no orphaned
    worker keeps generating.
    """
    _start_time = time.time()

    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail={"error": "Query cannot be empty"})
    if len(body.query) > get_config().max_query_length:
        raise HTTPException(
            status_code=400, detail={"error": "Query too long - maximum 2000 characters"}
        )

    session_id = body.session_id
    if session_id:
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(
                status_code=400, detail={"error": "session_id must be a valid UUID"}
            )
    else:
        session_id = str(uuid.uuid4())

    request_id = get_request_id()
    sink = ContentSink()
    # Control frames from the graph producer. The sink queue carries answer
    # deltas published by graph nodes; the producer pushes ("status"/"node",
    # ready-to-emit SSE string), ("final_state", state-dict) and
    # ("graph_error", ready-to-emit frame) tuples.
    frame_queue: "asyncio.Queue[tuple[str, Any]]" = asyncio.Queue()
    # Graph nodes publish answer deltas into the sink; forwarded here so the
    # SSE loop drains a single queue (content + control frames, in order).
    sink.subscribe(frame_queue)

    def sse(payload: dict) -> str:
        frame = dict(payload)
        frame.setdefault("request_id", request_id)
        return f"data: {json.dumps(frame)}\n\n"

    async def produce_graph_result() -> None:
        """Run the graph, forwarding node-complete frames into frame_queue.

        Answer deltas stream through the installed ContentSink; nodes pick it
        up via the contextvar (LangGraph sub-tasks inherit it at creation and
        share the sink object by reference).
        """
        token = install_content_sink(sink)
        token_scope = begin_token_scope()
        try:
            initial_state = create_initial_state(user_query=body.query, session_id=session_id)

            graph = get_graph(checkpointer=request.app.state.checkpointer)

            langsmith_config = langsmith_tracer.get_run_config(
                run_name=f"axiom-query-stream-{session_id[:8]}",
                session_id=session_id,
                metadata={
                    "user_query": body.query[:100],
                    "stub_mode": _compute_stub_mode(),
                },
            )
            full_config = {
                **langsmith_config,
                "configurable": {"thread_id": session_id},
                "recursion_limit": 50,
            }

            node_names = set(get_graph_node_names())
            seen_step_count = 0
            last_output: dict = {}
            generating_announced = False

            def node_frames(output: dict) -> list[str]:
                """New trace steps from one node output, as SSE frames."""
                nonlocal seen_step_count
                trace_steps = output.get("trace_steps", []) or []
                new_steps = trace_steps[seen_step_count:]
                seen_step_count = len(trace_steps)
                frames = []
                for step in new_steps:
                    step_data = (
                        step.model_dump()
                        if hasattr(step, "model_dump")
                        else (step if isinstance(step, dict) else {})
                    )
                    frames.append(sse({"type": "node_complete", "trace_step": step_data}))
                return frames

            timed_out = False
            try:
                async with asyncio.timeout(QUERY_GRAPH_TIMEOUT_SEC):
                    # Primary path: astream_events v2 — yields on_chain_end per node.
                    try:
                        async for event in graph.astream_events(
                            initial_state, full_config, version="v2"
                        ):
                            event_type = event.get("event", "")
                            metadata = event.get("metadata", {})
                            node_name = metadata.get("langgraph_node", "")

                            if not generating_announced and node_name == "generate_answer":
                                generating_announced = True
                                frame_queue.put_nowait(
                                    ("status", sse({"type": "status", "stage": "generating"}))
                                )

                            if event_type != "on_chain_end" or node_name not in node_names:
                                continue

                            output = event.get("data", {}).get("output", {})
                            if not isinstance(output, dict):
                                continue

                            last_output = output
                            for frame in node_frames(output):
                                frame_queue.put_nowait(("node", frame))
                    except asyncio.CancelledError:
                        raise
                    except Exception as stream_exc:
                        logger.warning(
                            "astream_events failed (%s), falling back to astream", stream_exc
                        )
                        seen_step_count = 0
                        last_output = {}
                        async for chunk in graph.astream(initial_state, full_config):
                            if not isinstance(chunk, dict):
                                continue
                            for node_name, state_update in chunk.items():
                                if node_name.startswith("__") or not isinstance(state_update, dict):
                                    continue
                                last_output = state_update
                                for frame in node_frames(state_update):
                                    frame_queue.put_nowait(("node", frame))
            except TimeoutError:
                # asyncio.timeout converts cancellation on budget exhaustion
                # into TimeoutError; a genuine mid-run CancelledError (client
                # disconnect) passes through unconverted.
                timed_out = True

            if timed_out:
                frame_queue.put_nowait(
                    (
                        "graph_error",
                        sse(
                            {
                                "type": "error",
                                "code": "query_timeout",
                                "message": "Query timed out. Try a simpler query.",
                            }
                        ),
                    )
                )
                return

            # After streaming, retrieve full final state from checkpointer
            try:
                snap = await graph.aget_state({"configurable": {"thread_id": session_id}})
                final_state = snap.values if snap and snap.values else last_output
            except Exception:
                final_state = last_output

            final_state["langsmith_trace_url"] = langsmith_tracer.get_trace_url(session_id)

            trace_steps_raw = final_state.get("trace_steps", []) or []
            _trace_store[session_id] = [
                step.model_dump() if hasattr(step, "model_dump") else dict(step)
                for step in trace_steps_raw
            ]
            _attach_trace_citations(_trace_store[session_id], _citations_from_state(final_state))
            await _persist_trace(session_id, _trace_store[session_id])

            frame_queue.put_nowait(("final_state", final_state))
        except Exception:
            logger.exception("Streaming query error (session=%s)", session_id)
            frame_queue.put_nowait(("graph_error", sse(sse_error_event(request_id=request_id))))
        finally:
            prompt_tokens, completion_tokens = end_token_scope(token_scope)
            if prompt_tokens or completion_tokens:
                observe_query_tokens("/api/query/stream", prompt_tokens, completion_tokens)
            reset_content_sink(token)

    async def event_stream():
        producer = asyncio.create_task(produce_graph_result())
        try:
            # Immediate stage signal: the pipeline begins with cache lookup
            # and retrieval before any text can stream.
            yield sse({"type": "status", "stage": "retrieving"})
            final_state: Optional[dict] = None
            graph_error_event: Optional[str] = None

            while True:
                try:
                    kind, payload = await asyncio.wait_for(frame_queue.get(), timeout=0.25)
                except TimeoutError:
                    # Bounded wait so a client disconnect is noticed even when
                    # the pipeline is quiet (retrieval can take seconds).
                    if await request.is_disconnected():
                        logger.info(
                            "SSE client disconnected (session=%s) — cancelling graph work",
                            session_id,
                        )
                        break
                    continue

                if kind == "content":
                    yield sse({"type": "content", "delta": payload})
                elif kind == "final_state":
                    final_state = payload
                    break
                elif kind == "graph_error":
                    graph_error_event = payload
                    break
                else:
                    # status / node_complete frames pass through as emitted
                    yield payload

            if final_state is not None:
                yield sse(_sources_from_state(final_state))

                ragas = final_state.get("ragas_scores")
                eval_mode = (
                    ragas.evaluation_mode
                    if ragas and hasattr(ragas, "evaluation_mode")
                    else "unknown"
                )

                response_obj = QueryResponse(
                    session_id=session_id,
                    final_answer=final_state.get("final_answer", ""),
                    confidence=_serialize_model(final_state.get("confidence")),
                    classification=_serialize_model(final_state.get("classification")),
                    retrieval_strategy=final_state.get("retrieval_strategy", ""),
                    ragas_scores=_serialize_model(ragas),
                    scores_history=[
                        _serialize_model(s) for s in final_state.get("scores_history", [])
                    ],
                    reranked_chunks=[
                        _serialize_model(c) for c in final_state.get("reranked_chunks", [])
                    ],
                    correction_attempts=final_state.get("correction_attempts", 0),
                    correction_history=[
                        _serialize_model(c) for c in final_state.get("correction_history", [])
                    ],
                    trace_steps=[_serialize_model(s) for s in final_state.get("trace_steps", [])],
                    served_from_cache=final_state.get("served_from_cache", False),
                    is_complete=final_state.get("is_complete", True),
                    error=final_state.get("error"),
                    total_latency_ms=round((time.time() - _start_time) * 1000, 2),
                    parallel_timing=_serialize_model(final_state.get("parallel_timing")),
                    cache_result=_serialize_model(final_state.get("cache_result")),
                    langsmith_trace_url=final_state.get("langsmith_trace_url"),
                    decomposed=final_state.get("decomposed", False),
                    sub_query_results=[
                        _serialize_model(r) for r in final_state.get("sub_query_results", [])
                    ],
                    evaluation_mode=eval_mode,
                    web_search_used=final_state.get("web_search_used", False),
                    web_search_chunks=final_state.get("web_search_chunks", []),
                    document_chunk_count=final_state.get("document_chunk_count", 0),
                    web_chunk_count=final_state.get("web_chunk_count", 0),
                    system_health=dict(_system_health),
                )

                yield sse({"type": "done", "result": response_obj.model_dump()})
                yield "data: [DONE]\n\n"
            elif graph_error_event is not None:
                yield graph_error_event
                yield "data: [DONE]\n\n"
            else:
                # Client disconnected mid-stream: stop without a terminal
                # frame — no one is left to read it. The producer task is
                # cancelled in the finally block below.
                pass

        except Exception:
            logger.exception("Streaming query error (session=%s)", session_id)
            yield sse(sse_error_event(request_id=request_id))
            yield "data: [DONE]\n\n"
        finally:
            # No orphaned workers: cancel graph work and wait for the
            # cancellation to land before the response finishes. The producer
            # never raises a non-cancellation exception (it reports failures
            # as graph_error frames), so this await is teardown only.
            if not producer.done():
                producer.cancel()
            try:
                await producer
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning("Graph producer teardown error (session=%s)", session_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


ACCEPTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@api_router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("5/minute")
async def ingest_document(request: Request, file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ACCEPTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={"error": "Unsupported file type. Accepted: pdf, txt, md"},
        )

    # Early rejection: check Content-Length header before buffering.
    # Most multipart clients send this. If present and over limit, reject
    # immediately with no memory allocation.
    _cl = request.headers.get("content-length")
    if _cl:
        try:
            if int(_cl) > get_config().max_ingest_size_mb * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": f"File too large. Maximum {get_config().max_ingest_size_mb}MB"
                    },
                )
        except ValueError:
            pass  # Malformed Content-Length — fall through to post-read check

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail={"error": "File is empty"})

    if len(content) > get_config().max_ingest_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail={"error": "File too large. Maximum 50MB"},
        )

    detected_type = magic.from_buffer(content, mime=True)
    allowed_mime_types = {
        "application/pdf",
        "text/plain",
        "text/markdown",
        "text/x-markdown",
    }
    if detected_type not in allowed_mime_types:
        raise HTTPException(
            status_code=415,
            detail={
                "error": f"Unsupported file type: {detected_type}. Allowed: PDF, TXT, Markdown."
            },
        )

    if _ingest_semaphore.locked():
        raise HTTPException(
            status_code=429,
            detail={"error": "Too many concurrent ingestion requests. Please try again shortly."},
        )
    await _ingest_semaphore.acquire()
    try:
        chunker = DocumentChunker()

        # One parse path for uploads AND connectors (Wave 4): digital PDFs
        # via pdfplumber, scanned PDFs via the optional Docling adapter when
        # enabled — otherwise a visible 422, never a silently shrunken doc.
        # Offloaded to a thread: OCR conversion is long-running and must not
        # pin the event loop.
        try:
            outcome = await asyncio.to_thread(
                parse_document, content, ext, ocr_enabled=get_config().ocr_enabled
            )
        except ScannedPdfNotSupportedError as exc:
            raise HTTPException(status_code=422, detail={"error": str(exc)})
        except OcrExtraMissingError as exc:
            # Server-side capability gap (the extra is not installed) — 503
            # with the install hint beats an opaque 500.
            raise HTTPException(status_code=503, detail={"error": str(exc)})

        chunks = chunker.chunk(outcome.pages, source=filename, origin_type="upload")

        indexer = get_dual_indexer()
        result = await indexer.index_chunks(chunks)

        if result.get("bm25") == "indexed" or result.get("vector") == "indexed":
            # The corpus changed — cached answers may reference stale content.
            # Cache entries carry no source lineage, so the only sound
            # invalidation is a full clear.
            cleared = await semantic_cache.clear()
            if cleared:
                logger.info(
                    "Cleared %d semantic cache entries after ingest of %s", cleared, filename
                )

        doc_id = _make_doc_id(filename)
        doc = {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_count": len(chunks),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "status": result.get("vector", "unknown"),
        }
        _ingested_docs.append(doc)
        await _persist_ingested_doc(doc_id, filename, len(chunks), len(content))

        return IngestResponse(
            filename=filename,
            chunk_count=len(chunks),
            status="indexed",
            doc_id=doc_id,
            mode=result.get("mode"),
            bm25=result.get("bm25"),
            vector=result.get("vector"),
            chunks=[
                {
                    "chunk_id": c["chunk_id"],
                    "source": c["source"],
                    "chunk_index": c["chunk_index"],
                    "token_count": c["token_count"],
                    "preview": c["content"][:100] + "..."
                    if len(c["content"]) > 100
                    else c["content"],
                }
                for c in chunks[:5]
            ],
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Ingest failed for %s", filename)
        raise HTTPException(
            status_code=500,
            detail=error_detail(
                INTERNAL_ERROR,
                GENERIC_INTERNAL_MESSAGE,
                filename=filename,
            ),
        )
    finally:
        _ingest_semaphore.release()


# ---------------------------------------------------------------------------
# Connector runs (Wave 4, D3/D4): background ingestion with a status lifecycle.
#
# Runs execute via FastAPI BackgroundTasks (no queue infrastructure). Each run
# is the batch boundary: every fetched document is parsed and chunked into one
# list, indexed with ONE index_run call, and the semantic cache is cleared
# ONCE. Per-document status, failures, and parse confidence are recorded on
# the run and persisted through _persist_ingested_doc — nothing fails
# silently, and no run reports success while having indexed nothing.
# ---------------------------------------------------------------------------

_connector_runs: Dict[str, Dict[str, Any]] = {}
_CONNECTOR_RUN_TERMINAL = ("completed", "partial", "failed")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prune_connector_runs() -> None:
    """Drop terminal runs older than the retention window (bounded memory)."""
    retention = get_config().connector_run_retention_seconds
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=retention)).isoformat()
    for run_id in list(_connector_runs):
        run = _connector_runs[run_id]
        if run.get("status") in _CONNECTOR_RUN_TERMINAL:
            completed_at = str(run.get("completed_at") or "")
            if completed_at and completed_at < cutoff:
                _connector_runs.pop(run_id, None)


async def _crawl_dedup_store() -> CrawlDedupStore:
    """Redis-backed crawl dedup under axiom:crawl:, with in-process fallback."""
    try:
        import redis.asyncio as aioredis

        cfg = get_config()
        client = aioredis.Redis(
            host=cfg.redis_host,
            port=cfg.redis_port,
            password=cfg.redis_password or None,
            decode_responses=True,
        )
        # redis-py types ping as sync-or-await depending on the client flavor;
        # this client is asyncio, but the stub union forces a runtime narrow.
        ping = client.ping()
        if inspect.isawaitable(ping):
            await ping
        return CrawlDedupStore(redis_client=client)
    except Exception as exc:
        logger.warning("Crawl dedup: Redis unavailable (%s) — using in-process dedup", exc)
        return CrawlDedupStore()


async def _fetch_connector_documents(
    connector: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Fetch raw documents from a connector, normalized for the parse seam.

    Each item: {source, filename, content, ext, fetched_at, content_hash,
    size}. Per-object failures are sanitized and returned in ``errors``.
    """
    if connector == "s3":
        objects, errors = await fetch_s3_objects()
        normalized = [
            {
                "source": obj.source,
                "filename": obj.filename,
                "content": obj.content,
                "ext": os.path.splitext(obj.filename)[1].lower(),
                "fetched_at": obj.fetched_at,
                "content_hash": obj.content_hash,
                "size": obj.size,
            }
            for obj in objects
        ]
        return normalized, errors

    seeds = parse_seeds(get_config().crawl_seeds)
    dedup = await _crawl_dedup_store()
    fetched, errors = await crawl_pages(seeds, dedup=dedup)
    normalized = [
        {
            "source": page.source,
            "filename": page.filename,
            "content": page.content,
            "ext": page.ext,
            "fetched_at": page.fetched_at,
            "content_hash": page.content_hash,
            "size": page.size,
        }
        for page in fetched
    ]
    return normalized, errors


async def _ingest_fetched_document(
    item: Dict[str, Any],
    connector: str,
    chunks_all: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Parse, chunk, and bookkeep one fetched document; return its run row.

    Chunked chunks are appended to ``chunks_all`` for the run-level batch
    index; this function itself never touches the indexers or the cache.
    """
    source = str(item["source"])
    row: Dict[str, Any] = {
        "source": source,
        "origin_type": connector,
        "origin_uri": source,
        "fetched_at": item["fetched_at"].isoformat(),
        "content_hash": item["content_hash"],
        "size": item["size"],
        "chunk_count": 0,
        "status": "completed",
        "parse_confidence": None,
        "error": None,
    }

    try:
        outcome = await asyncio.to_thread(
            parse_document,
            item["content"],
            str(item["ext"]),
            ocr_enabled=get_config().ocr_enabled,
        )
    except ScannedPdfNotSupportedError as exc:
        row["status"] = "failed"
        row["error"] = str(exc)[:200]
        return row
    except OcrExtraMissingError as exc:
        row["status"] = "failed"
        row["error"] = str(exc)[:200]
        return row
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = f"Parsing failed: {str(exc)[:160]}"
        return row

    chunker = DocumentChunker()
    chunks = chunker.chunk(outcome.pages, source=source, origin_type=connector)
    if not chunks:
        row["status"] = "failed"
        row["error"] = "No indexable content after parsing and chunking."
        return row

    row["chunk_count"] = len(chunks)
    row["parse_confidence"] = outcome.parse_confidence
    chunks_all.extend(chunks)
    return row


async def _execute_connector_run(run_id: str, connector: str) -> None:
    """Background executor for one connector run (FastAPI BackgroundTasks)."""
    run = _connector_runs.get(run_id)
    if run is None:
        return
    run["status"] = "running"
    errors: List[Dict[str, str]] = []
    chunks_all: List[Dict[str, Any]] = []

    try:
        try:
            fetched, errors = await _fetch_connector_documents(connector)
        except S3ConnectorError as exc:
            run["status"] = "failed"
            run["completed_at"] = _now_iso()
            run["errors"] = [{"source": "s3", "reason": str(exc)}]
            return

        run["errors"] = list(errors)
        for item in fetched:
            row = await _ingest_fetched_document(item, connector, chunks_all)
            run["documents"].append(row)

        if chunks_all:
            index_result = await get_dual_indexer().index_run(chunks_all)
            run["indexed_chunks"] = index_result.get("chunk_count")
            if "indexed" in (index_result.get("vector"), index_result.get("bm25")):
                # The index contents changed (either component): cached answers
                # may reference stale retrieval state, so the cache clears
                # even when the other component failed.
                cleared = await semantic_cache.clear()
                run["cache_cleared"] = True
                logger.info(
                    "Connector run %s indexed %s chunks across %s sources; cleared %s cache entries",
                    run_id,
                    index_result.get("chunk_count"),
                    index_result.get("sources"),
                    cleared,
                )
            # Partial or full indexing failure is surfaced in the run record —
            # never a silent empty-but-successful outcome.
            failures = []
            if index_result.get("vector") != "indexed":
                detail = index_result.get("vector_error") or index_result.get("vector") or "not run"
                failures.append(f"vector: {str(detail)[:120]}")
            if index_result.get("bm25") != "indexed":
                failures.append(f"bm25: {index_result.get('bm25', 'not run')}")
            if failures:
                run["errors"].append({
                    "source": "indexing",
                    "reason": f"Indexing did not complete ({'; '.join(failures)})",
                })

        completed = [d for d in run["documents"] if d["status"] == "completed"]
        failed = [d for d in run["documents"] if d["status"] != "completed"]
        if not completed and (failed or run["errors"]):
            run["status"] = "failed"
        elif failed or run["errors"]:
            run["status"] = "partial"
        else:
            run["status"] = "completed"
    except Exception as exc:
        logger.exception("Connector run %s failed", run_id)
        run["status"] = "failed"
        run["errors"].append({"source": "run", "reason": f"Run failed: {str(exc)[:160]}"})
    finally:
        run["completed_at"] = _now_iso()

        for doc in run["documents"]:
            await _persist_ingested_doc(
                doc_id=_make_doc_id(doc["source"]),
                filename=doc["source"],
                chunk_count=doc["chunk_count"],
                file_size_bytes=doc["size"] or 0,
                origin_type=doc["origin_type"],
                origin_uri=doc["origin_uri"],
                fetched_at=datetime.fromisoformat(doc["fetched_at"]),
                content_hash=doc["content_hash"],
                status=doc["status"],
                error_reason=doc["error"],
                parse_confidence=doc["parse_confidence"],
            )
        _prune_connector_runs()


@api_router.post(
    "/connectors/{connector}/run",
    response_model=ConnectorRunResponse,
    status_code=202,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("5/minute")
async def start_connector_run(request: Request, connector: str, background_tasks: BackgroundTasks) -> ConnectorRunResponse:
    """Start a background connector run; poll it at /connectors/runs/{run_id}.

    Returns 503 when the connector is unconfigured — an explicit user request
    must fail visibly, unlike the automatic optional-disabled fallbacks.
    """
    if connector == "s3":
        if not is_s3_configured():
            raise HTTPException(
                status_code=503,
                detail=error_detail(
                    "CONNECTOR_NOT_CONFIGURED",
                    "S3 connector is not configured. Set S3_BUCKET (plus credentials) to enable it.",
                ),
            )
        config_summary: Dict[str, Any] = s3_run_config_summary()
    elif connector == "crawl":
        if not is_crawl_configured():
            raise HTTPException(
                status_code=503,
                detail=error_detail(
                    "CONNECTOR_NOT_CONFIGURED",
                    "Web-crawl connector is not configured. Set CRAWL_SEEDS to enable it.",
                ),
            )
        config_summary = crawl_run_config_summary()
    else:
        raise HTTPException(
            status_code=404,
            detail=error_detail("CONNECTOR_NOT_FOUND", "Unknown connector: available connectors are 's3' and 'crawl'."),
        )

    run_id = uuid.uuid4().hex[:12]
    _connector_runs[run_id] = {
        "run_id": run_id,
        "connector": connector,
        "status": "pending",
        "started_at": _now_iso(),
        "completed_at": None,
        "documents": [],
        "errors": [],
        "config": config_summary,
        "indexed_chunks": None,
        "cache_cleared": False,
    }
    _prune_connector_runs()
    background_tasks.add_task(_execute_connector_run, run_id, connector)
    return ConnectorRunResponse(run_id=run_id, connector=connector, status="pending")


@api_router.get(
    "/connectors/runs/{run_id}",
    response_model=ConnectorRunStatusResponse,
    dependencies=[Depends(require_api_key)],
)
async def get_connector_run(request: Request, run_id: str) -> ConnectorRunStatusResponse:
    """Authenticated polling endpoint for a background connector run."""
    run = _connector_runs.get(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=error_detail("RUN_NOT_FOUND", "No connector run with the given id (runs are cleared after the retention window)."),
        )
    return ConnectorRunStatusResponse(**run)


async def _find_doc_by_id(doc_id: str) -> dict[str, Any] | None:
    """Locate an ingested document by doc_id — in-memory first, then PostgreSQL."""
    doc = next((d for d in _ingested_docs if d.get("doc_id") == doc_id), None)
    if doc:
        return doc
    if not get_engine():
        return None
    try:
        from sqlalchemy import text as sa_text

        async with get_engine().connect() as conn:
            row = await conn.execute(
                sa_text(
                    "SELECT doc_id, filename, chunk_count, file_size_bytes "
                    "FROM ingested_documents WHERE doc_id = :did"
                ),
                {"did": doc_id},
            )
            result = row.fetchone()
            if result:
                return dict(result._mapping)
    except Exception as exc:
        logger.warning("Failed to look up doc %s: %s", doc_id, exc)
    return None


@api_router.delete(
    "/documents/{doc_id}",
    response_model=DeleteDocumentResponse,
    dependencies=[Depends(require_api_key)],
)
async def delete_document(doc_id: str):
    """Delete a document and all of its chunk embeddings (pgvector + BM25 + lineage).

    Note: chunk identity is per-source (filename), so deleting any lineage
    record for a source purges that source's chunks — the latest ingest owns
    the content for its filename.
    """
    doc = await _find_doc_by_id(doc_id)
    if not doc:
        raise HTTPException(
            status_code=404, detail={"error": "Document not found", "doc_id": doc_id}
        )

    filename = doc["filename"]
    try:
        deleted_chunks = await vector_store.delete_by_source(filename)
        await bm25_index.remove_source(filename)
        await _delete_ingested_doc_record(doc_id)
    except Exception:
        logger.exception("Delete failed for doc %s (%s)", doc_id, filename)
        raise HTTPException(
            status_code=500,
            detail=error_detail(
                INTERNAL_ERROR,
                "Failed to delete the document. Check server logs for details.",
                doc_id=doc_id,
            ),
        )

    cache_keys_cleared = await semantic_cache.clear()

    # Remove in-memory lineage last so a failed store delete leaves the
    # record queryable for a retry.
    _ingested_docs[:] = [d for d in _ingested_docs if d.get("doc_id") != doc_id]

    return DeleteDocumentResponse(
        doc_id=doc_id,
        filename=filename,
        deleted_chunks=deleted_chunks,
        cache_keys_cleared=cache_keys_cleared,
        status="deleted",
    )


@api_router.get(
    "/trace/{session_id}", response_model=TraceResponse, dependencies=[Depends(require_api_key)]
)
async def get_trace(session_id: str):
    # Postgres is the source of truth — any worker can serve any trace.
    # The in-process store is a degraded-mode fallback (no DB, or a row the
    # DB lost via a failed write-through).
    trace_steps = await _load_trace(session_id)
    if trace_steps is None:
        trace_steps = _trace_store.get(session_id, [])

    if not trace_steps:
        raise HTTPException(
            status_code=404,
            detail={"error": f"No trace found for session {session_id}", "session_id": session_id},
        )

    return TraceResponse(session_id=session_id, trace_steps=trace_steps)


async def _pg_table_count(table: str) -> Optional[int]:
    """Row count from PostgreSQL; None when the store is unavailable.

    ``table`` is caller-controlled, so it is checked against the small set of
    app tables instead of being interpolated blindly.
    """
    if table not in {"pipeline_traces", "ingested_documents", "eval_runs"}:
        return None
    engine = get_engine()
    if engine is None:
        return None
    try:
        from sqlalchemy import text as sa_text

        async with engine.connect() as conn:
            result = await conn.execute(sa_text(f"SELECT COUNT(*) FROM {table}"))
            return int(result.scalar() or 0)
    except Exception as exc:
        logger.warning("Failed to count %s: %s", table, exc)
        return None


@api_router.get("/citations/{trace_id}", dependencies=[Depends(require_api_key)])
async def get_citations(trace_id: str):
    """Chunk-level citations persisted for a historical trace (history replay).

    Citations ride the trace's JSONB (rerank/check_cache step detail), written
    at trace-persist time. Traces persisted before this enrichment come back
    with an empty citation list; live responses carry reranked_chunks directly
    and do not need this endpoint.
    """
    trace_steps = await _load_trace(trace_id)
    if trace_steps is None:
        trace_steps = _trace_store.get(trace_id, [])

    if not trace_steps:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"No trace found for session {trace_id}",
                "session_id": trace_id,
            },
        )

    return {
        "session_id": trace_id,
        "citations": _extract_trace_citations(trace_steps if isinstance(trace_steps, list) else []),
    }


@api_router.post("/feedback", dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def post_feedback(request: Request, body: FeedbackRequest):
    """Record thumbs-up/down feedback on a query trace (persistence + visibility).

    Automatic strategy re-tuning from this signal is a follow-up wave — this
    endpoint only persists and exposes it.
    """
    trace_id = body.trace_id.strip()
    if not trace_id:
        raise HTTPException(status_code=400, detail={"error": "trace_id cannot be empty"})

    engine = get_engine()
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Feedback storage is not available",
                "guidance": "Feedback persistence requires PostgreSQL. Check the database connection and retry.",
            },
        )

    try:
        from sqlalchemy import text as sa_text

        async with engine.begin() as conn:
            trace_row = await conn.execute(
                sa_text("SELECT 1 FROM pipeline_traces WHERE session_id = :tid"),
                {"tid": trace_id},
            )
            if trace_row.fetchone() is None:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": f"No trace found for session {trace_id}",
                        "session_id": trace_id,
                    },
                )
            row = (
                await conn.execute(
                    sa_text("""
                INSERT INTO query_feedback (trace_id, rating, comment, query_snippet)
                VALUES (:tid, :rating, :comment, :snippet)
                RETURNING id, created_at
            """),
                    {
                        "tid": trace_id,
                        "rating": body.rating,
                        "comment": body.comment.strip()
                        if body.comment and body.comment.strip()
                        else None,
                        "snippet": body.query_snippet,
                    },
                )
            ).fetchone()
    except HTTPException:
        # Intentional envelopes (unknown trace 404) pass through untouched.
        raise
    except Exception:
        logger.exception("Feedback recording failed (trace=%s)", trace_id)
        raise HTTPException(
            status_code=500,
            detail=error_detail(
                INTERNAL_ERROR,
                GENERIC_INTERNAL_MESSAGE,
                session_id=trace_id,
            ),
        )

    created_at = row[1]
    return {
        "id": row[0],
        "trace_id": trace_id,
        "rating": body.rating,
        "comment": body.comment,
        "created_at": created_at.isoformat()
        if hasattr(created_at, "isoformat")
        else str(created_at),
        "status": "recorded",
    }


@api_router.get("/feedback/summary", dependencies=[Depends(require_api_key)])
async def feedback_summary():
    """Aggregate feedback: counts per rating plus the most recent items.

    Read-only surface for the eval dashboard; degrades to zero counts when
    PostgreSQL is unavailable (the write path, unlike this, refuses 503).
    """
    counts = {"up": 0, "down": 0}
    recent: List[Dict[str, Any]] = []
    engine = get_engine()
    if engine is not None:
        try:
            from sqlalchemy import text as sa_text

            async with engine.connect() as conn:
                totals = (
                    await conn.execute(
                        sa_text(
                            "SELECT COALESCE(SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END), 0) AS up, "
                            "COALESCE(SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END), 0) AS down "
                            "FROM query_feedback"
                        )
                    )
                ).fetchone()
                counts["up"] = int(totals[0])
                counts["down"] = int(totals[1])
                rows = (
                    await conn.execute(
                        sa_text(
                            "SELECT id, trace_id, rating, comment, query_snippet, created_at "
                            "FROM query_feedback ORDER BY created_at DESC LIMIT 20"
                        )
                    )
                ).fetchall()
                for r in rows:
                    recent.append(
                        {
                            "id": r[0],
                            "trace_id": r[1],
                            "rating": int(r[2]),
                            "comment": r[3],
                            "query_snippet": r[4],
                            "created_at": r[5].isoformat()
                            if hasattr(r[5], "isoformat")
                            else str(r[5]),
                        }
                    )
        except Exception as exc:
            logger.warning("Failed to aggregate feedback summary: %s", exc)

    return {
        "total": counts["up"] + counts["down"],
        "counts": counts,
        "recent": recent,
    }


@api_router.get("/eval/runs", dependencies=[Depends(require_api_key)])
async def list_eval_runs():
    """List persisted eval runs (newest first) for the /eval dashboard.

    Reads the PG-backed eval_runs table written by the background eval runner.
    This closes the runs-list gap: GET /eval/results remains a single-worker
    local-file read — fine for one-process dev, fragile for multi-worker
    deployments (documented limitation, not fixed this wave). Without
    PostgreSQL the list is empty.
    """
    engine = get_engine()
    if engine is None:
        return {"runs": [], "count": 0}

    runs: List[Dict[str, Any]] = []
    try:
        from sqlalchemy import text as sa_text

        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    sa_text(
                        "SELECT job_id, status, progress, total, aggregate, error, started_at, completed_at "
                        "FROM eval_runs ORDER BY started_at DESC NULLS LAST LIMIT 100"
                    )
                )
            ).fetchall()
            for r in rows:
                runs.append(
                    {
                        "job_id": r[0],
                        "status": r[1],
                        "progress": r[2],
                        "total": r[3],
                        "aggregate": _json_or_none(r[4]),
                        "error": r[5],
                        "started_at": r[6].isoformat() if hasattr(r[6], "isoformat") else str(r[6]),
                        "completed_at": r[7].isoformat()
                        if hasattr(r[7], "isoformat")
                        else str(r[7]),
                    }
                )
    except Exception as exc:
        logger.warning("Failed to list eval runs: %s", exc)

    return {"runs": runs, "count": len(runs)}


@api_router.get("/stats", dependencies=[Depends(require_api_key)])
async def get_stats():
    pg_connected = await vector_store.is_connected()
    cache_stats = await semantic_cache.stats()
    pg_docs = await _pg_table_count("ingested_documents")
    pg_traces = await _pg_table_count("pipeline_traces")
    return {
        # Postgres counts when available — all workers agree; in-process
        # fallbacks only when the store is down.
        "indexed_documents": pg_docs if pg_docs is not None else len(_ingested_docs),
        "bm25_doc_count": bm25_index.count(),
        "vector_doc_count": await vector_store.count() if pg_connected else 0,
        "cache_entries": cache_stats["total_entries"],
        "cache_hits": cache_stats["total_hits"],
        "total_queries_processed": pg_traces if pg_traces is not None else len(_trace_store),
        "stub_mode": _compute_stub_mode(),
    }


async def _run_eval_background(job_id: str) -> None:
    """Execute the benchmark in-process; updates _eval_jobs for polling."""
    from axiom.eval_suite.benchmark import BENCHMARK_QUERIES
    from axiom.eval_suite.runner import EvalRunner

    runner = EvalRunner()
    suite_start = time.perf_counter()
    results: List[Dict[str, Any]] = []

    try:
        await runner._ensure_services()
        _eval_jobs[job_id]["total"] = len(BENCHMARK_QUERIES)
        await _upsert_eval_run(job_id, _eval_jobs[job_id])

        for i, bq in enumerate(BENCHMARK_QUERIES):
            session_id = f"eval-{job_id}-{i:02d}"
            res = await runner.run_single(bq, session_id)
            results.append(res)
            _eval_jobs[job_id]["progress"] = i + 1
            _eval_jobs[job_id]["latest"] = res
            _eval_jobs[job_id]["results"] = list(results)
            await _upsert_eval_run(job_id, _eval_jobs[job_id])

        total_s = time.perf_counter() - suite_start
        runner.results = results
        aggregate = runner._compute_aggregate(total_s, results)
        runner.save_results(aggregate)

        _eval_jobs[job_id]["status"] = "complete"
        _eval_jobs[job_id]["aggregate"] = aggregate
        _eval_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        await _upsert_eval_run(job_id, _eval_jobs[job_id])
    except Exception as exc:
        logger.exception("Eval job %s failed: %s", job_id, exc)
        _eval_jobs[job_id]["status"] = "failed"
        # Sanitized envelope message, not str(exc): the status endpoint serves
        # this field verbatim (and eval_runs.error is a TEXT column). Full
        # detail is in the exception log above.
        _eval_jobs[job_id]["error"] = GENERIC_INTERNAL_MESSAGE
        _eval_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        await _upsert_eval_run(job_id, _eval_jobs[job_id])


async def _run_eval_with_semaphore(job_id: str) -> None:
    async with _eval_semaphore:
        await _run_eval_background(job_id)


@api_router.post("/eval/run", dependencies=[Depends(require_api_key)])
@limiter.limit("2/hour")
async def run_eval_suite(request: Request, background_tasks: BackgroundTasks):
    """Start the 30-query benchmark in the background. Poll GET /api/eval/status/{job_id}."""
    if _eval_semaphore.locked():
        raise HTTPException(
            status_code=409, detail={"error": "An evaluation run is already in progress."}
        )

    job_id = uuid.uuid4().hex[:8]
    from axiom.eval_suite.benchmark import BENCHMARK_QUERIES

    _eval_jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "total": len(BENCHMARK_QUERIES),
        "results": [],
        "latest": None,
        "aggregate": None,
        "error": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    # The row lands before the 202-style response returns, so a poll from
    # another worker immediately sees the job.
    await _upsert_eval_run(job_id, _eval_jobs[job_id])
    background_tasks.add_task(_run_eval_with_semaphore, job_id)

    return {
        "job_id": job_id,
        "status": "started",
        "poll_url": f"/api/eval/status/{job_id}",
        "message": "Poll poll_url until status is complete; results are written to eval_results.json on success.",
    }


@api_router.get("/eval/status/{job_id}", dependencies=[Depends(require_api_key)])
async def eval_job_status(job_id: str):
    """Live progress for a benchmark job started via POST /api/eval/run.

    State lives in PostgreSQL, so any worker can serve any job's status —
    the caller is not pinned to the worker that started the run.
    """
    job = await _load_eval_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "Job not found", "job_id": job_id})
    return {"job_id": job_id, **job}


@api_router.post("/eval/run/stream", dependencies=[Depends(require_api_key)])
async def run_eval_suite_stream():
    """Stream SSE progress after each query (optional; use curl -N). Saves results on success."""
    from axiom.eval_suite.benchmark import BENCHMARK_QUERIES
    from axiom.eval_suite.runner import EvalRunner

    async def generate():
        runner = EvalRunner()
        suite_start = time.perf_counter()
        results: List[Dict[str, Any]] = []
        try:
            await runner._ensure_services()
            for i, bq in enumerate(BENCHMARK_QUERIES):
                session_id = f"eval-stream-{uuid.uuid4().hex[:6]}-{i:02d}"
                result = await runner.run_single(bq, session_id)
                results.append(result)
                rs = result.get("ragas_scores") or {}
                progress = {
                    "progress": f"{i + 1}/{len(BENCHMARK_QUERIES)}",
                    "query": bq["query"][:50],
                    "complete": result.get("is_complete"),
                    "strategy": result.get("actual_strategy"),
                    "faithfulness": rs.get("faithfulness"),
                    "error": result.get("error"),
                }
                yield f"data: {json.dumps(progress)}\n\n"

            total_s = time.perf_counter() - suite_start
            runner.results = results
            aggregate = runner._compute_aggregate(total_s, results)
            runner.save_results(aggregate)
            yield f"data: {json.dumps({'final': True, 'aggregate': aggregate})}\n\n"
        except Exception:
            logger.exception("Eval stream failed")
            yield f"data: {json.dumps(sse_error_event())}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api_router.get("/eval/results", dependencies=[Depends(require_api_key)])
async def get_eval_results():
    """Return the last saved eval_results.json if it exists."""
    import json

    results_path = Path(__file__).parent / "eval_results.json"
    if not results_path.exists():
        raise HTTPException(
            status_code=404,
            detail={"error": "No eval results found. Run POST /api/eval/run first."},
        )
    with open(results_path) as f:
        return json.load(f)


@api_router.get("/session/{session_id}/state", dependencies=[Depends(require_api_key)])
async def get_session_state(session_id: str, request: Request):
    """Return the last checkpointed state for a session."""
    try:
        graph = get_graph(checkpointer=request.app.state.checkpointer)
        config = {"configurable": {"thread_id": session_id}}
        state = await graph.aget_state(config)
        if state is None or state.values is None:
            raise HTTPException(status_code=404, detail={"error": "No state found for session"})
        return {
            "session_id": session_id,
            "has_state": True,
            "is_complete": state.values.get("is_complete", False),
            "correction_attempts": state.values.get("correction_attempts", 0),
            "retrieval_strategy": state.values.get("retrieval_strategy", ""),
            "decomposed": state.values.get("decomposed", False),
            "served_from_cache": state.values.get("served_from_cache", False),
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Session state lookup failed for %s", session_id)
        raise HTTPException(
            status_code=500,
            detail=error_detail(
                INTERNAL_ERROR,
                GENERIC_INTERNAL_MESSAGE,
                session_id=session_id,
            ),
        )


app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
