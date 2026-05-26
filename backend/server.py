"""AXIOM FastAPI Backend - Main API Server."""

import asyncio
import hashlib
import json
import logging
import os
import uuid
import time

import magic
from contextlib import asynccontextmanager, AsyncExitStack
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Request, BackgroundTasks, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s — %(message)s")

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / '.env')

logger = logging.getLogger(__name__)

QUERY_GRAPH_TIMEOUT_SEC = float(os.environ.get("QUERY_GRAPH_TIMEOUT_SEC", "180"))

from axiom.graph.graph import get_graph, get_graph_node_names
from axiom.graph.state import create_initial_state
from axiom.ingest.loader import DocumentChunker
from axiom.ingest.indexer import get_dual_indexer
from axiom.retrieval.bm25_index import bm25_index
from axiom.retrieval.vector_store import vector_store, get_engine
from axiom.retrieval.reranker import get_reranker
from axiom.cache.semantic_cache import semantic_cache
from axiom.evaluation.critic_llm import critic_llm
from axiom.evaluation.claude_evaluator import claude_evaluator
from axiom.observability.langsmith import langsmith_tracer
from axiom.config import get_config


# System health state. Populated during lifespan startup.
# Feature 3 will extend this to all 5 components and expose it in every API response.
_system_health: dict = {
    "pgvector": "unknown",
    "redis": "unknown",
    "reranker": "unknown",
    "web_search": "unknown",
    "evaluator": "unknown",
}

limiter = Limiter(key_func=get_remote_address)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    key: str = Security(_api_key_header),
) -> None:
    cfg = get_config()
    if not cfg.api_key:
        return
    if key != cfg.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app):
    checkpointer_stack = AsyncExitStack()
    await checkpointer_stack.__aenter__()
    app.state._checkpointer_stack = checkpointer_stack

    connected = await vector_store.connect()
    if connected:
        logger.info("pgvector connected — chunk_embeddings table ready")
        await _create_persistence_tables()
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

    get_reranker().load()
    _system_health["reranker"] = "loaded" if get_reranker().is_loaded() else "not_loaded"
    logger.info("Reranker: %s", _system_health["reranker"])

    from axiom.search.web_search import is_tavily_configured
    _system_health["web_search"] = "tavily" if is_tavily_configured() else "not_configured"
    logger.info("Web search: %s", _system_health["web_search"])

    logger.info(
        "System health at startup: pgvector=%s redis=%s reranker=%s "
        "web_search=%s evaluator=%s",
        _system_health["pgvector"],
        _system_health["redis"],
        _system_health["reranker"],
        _system_health["web_search"],
        _system_health["evaluator"],
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
        logger.info("Shutdown: closing Ollama httpx client...")
        try:
            if critic_llm._client:
                await critic_llm._client.aclose()
                logger.info("Shutdown: Ollama httpx client closed")
        except Exception as exc:
            logger.warning("Shutdown: Ollama httpx cleanup error: %s", exc)

    logger.info("Shutdown: cleanup complete")


async def _create_persistence_tables():
    """Create tables for traces, ingested docs, and eval runs if they don't exist."""
    try:
        from sqlalchemy import text as sa_text
        async with get_engine().begin() as conn:
            await conn.execute(sa_text("""
                CREATE TABLE IF NOT EXISTS pipeline_traces (
                    session_id TEXT PRIMARY KEY,
                    trace_data JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            await conn.execute(sa_text("""
                CREATE TABLE IF NOT EXISTS ingested_documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT,
                    chunk_count INTEGER,
                    file_size_bytes INTEGER,
                    indexed_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            await conn.execute(sa_text("""
                CREATE TABLE IF NOT EXISTS eval_runs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT,
                    progress INTEGER,
                    total INTEGER,
                    aggregate JSONB,
                    results JSONB,
                    started_at TIMESTAMPTZ DEFAULT NOW(),
                    completed_at TIMESTAMPTZ
                )
            """))
        logger.info("Persistence tables ready (pipeline_traces, ingested_documents, eval_runs)")
    except Exception as exc:
        logger.warning("Failed to create persistence tables: %s", exc)


async def _hydrate_ingested_docs():
    """Populate the in-memory _ingested_docs list from PostgreSQL on startup."""
    try:
        from sqlalchemy import text as sa_text
        async with get_engine().connect() as conn:
            rows = await conn.execute(sa_text(
                "SELECT doc_id, filename, chunk_count, file_size_bytes, indexed_at "
                "FROM ingested_documents ORDER BY indexed_at"
            ))
            for r in rows:
                row = dict(r._mapping)
                _ingested_docs.append({
                    "filename": row["filename"],
                    "chunk_count": row["chunk_count"],
                    "indexed_at": row["indexed_at"].isoformat() if hasattr(row["indexed_at"], "isoformat") else str(row["indexed_at"]),
                    "status": "indexed",
                })
        if _ingested_docs:
            logger.info("Hydrated %d ingested document records from PostgreSQL", len(_ingested_docs))
    except Exception as exc:
        logger.warning("Failed to hydrate ingested docs: %s", exc)


async def _persist_trace(session_id: str, trace_data: list) -> None:
    """Upsert a trace to PostgreSQL (write-through)."""
    if not get_engine():
        return
    try:
        from sqlalchemy import text as sa_text
        async with get_engine().begin() as conn:
            await conn.execute(sa_text("""
                INSERT INTO pipeline_traces (session_id, trace_data)
                VALUES (:sid, :data)
                ON CONFLICT (session_id) DO UPDATE SET trace_data = :data, created_at = NOW()
            """), {"sid": session_id, "data": json.dumps(trace_data)})
    except Exception as exc:
        logger.warning("Failed to persist trace %s: %s", session_id, exc)


async def _load_trace(session_id: str) -> list | None:
    """Load a trace from PostgreSQL. Returns None if not found."""
    if not get_engine():
        return None
    try:
        from sqlalchemy import text as sa_text
        async with get_engine().connect() as conn:
            row = await conn.execute(sa_text(
                "SELECT trace_data FROM pipeline_traces WHERE session_id = :sid"
            ), {"sid": session_id})
            result = row.fetchone()
            if result:
                return result[0]
        return None
    except Exception as exc:
        logger.warning("Failed to load trace %s: %s", session_id, exc)
        return None


async def _persist_ingested_doc(filename: str, chunk_count: int, file_size_bytes: int) -> None:
    """Insert an ingested document record to PostgreSQL."""
    if not get_engine():
        return
    try:
        from sqlalchemy import text as sa_text
        doc_id = hashlib.sha256(f"{filename}:{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:16]
        async with get_engine().begin() as conn:
            await conn.execute(sa_text("""
                INSERT INTO ingested_documents (doc_id, filename, chunk_count, file_size_bytes)
                VALUES (:did, :fn, :cc, :fsb)
                ON CONFLICT (doc_id) DO NOTHING
            """), {"did": doc_id, "fn": filename, "cc": chunk_count, "fsb": file_size_bytes})
    except Exception as exc:
        logger.warning("Failed to persist ingested doc %s: %s", filename, exc)


async def _persist_eval_run(job_id: str, job_data: dict) -> None:
    """Write final eval run result to PostgreSQL on completion."""
    if not get_engine():
        return
    try:
        from sqlalchemy import text as sa_text
        async with get_engine().begin() as conn:
            await conn.execute(sa_text("""
                INSERT INTO eval_runs (job_id, status, progress, total, aggregate, results, started_at, completed_at)
                VALUES (:jid, :st, :pr, :tot, :agg, :res, :sa, :ca)
                ON CONFLICT (job_id) DO UPDATE SET
                    status = :st, progress = :pr, aggregate = :agg, results = :res, completed_at = :ca
            """), {
                "jid": job_id,
                "st": job_data.get("status"),
                "pr": job_data.get("progress", 0),
                "tot": job_data.get("total", 0),
                "agg": json.dumps(job_data.get("aggregate")),
                "res": json.dumps(job_data.get("results")),
                "sa": job_data.get("started_at"),
                "ca": job_data.get("completed_at"),
            })
    except Exception as exc:
        logger.warning("Failed to persist eval run %s: %s", job_id, exc)


async def _hydrate_bm25_from_pgvector():
    """Load chunks from pgvector into in-memory BM25 so both indexes stay in sync."""
    try:
        from sqlalchemy import text as sa_text
        async with get_engine().connect() as conn:
            rows = await conn.execute(sa_text(
                "SELECT chunk_id, source, content, chunk_index, token_count FROM chunk_embeddings"
            ))
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
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(','),
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
    mode: Optional[str] = None
    bm25: Optional[str] = None
    vector: Optional[str] = None
    chunks: Optional[List[Dict[str, Any]]] = None


class TraceResponse(BaseModel):
    session_id: str
    trace_steps: List[Dict[str, Any]]


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
    return not pgvector_ok or not evaluator_ok or not reranker_ok


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
            "web_search": _system_health.get("web_search", "unknown"),
            "reranker": _system_health.get("reranker", "unknown"),
        },
        "langsmith": "enabled" if langsmith_tracer.is_enabled() else "disabled (set LANGCHAIN_TRACING_V2=true)",
        "checkpointing": "enabled (MemorySaver)",
    }


@api_router.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def process_query(request: Request, body: QueryRequest):
    _start_time = time.time()

    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail={"error": "Query cannot be empty"})
    if len(body.query) > get_config().max_query_length:
        raise HTTPException(status_code=400, detail={"error": "Query too long — maximum 2000 characters"})

    session_id = body.session_id
    if session_id:
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="session_id must be a valid UUID"
            )
    else:
        session_id = str(uuid.uuid4())
    current_node = None

    try:
        initial_state = create_initial_state(
            user_query=body.query,
            session_id=session_id
        )

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
            final_state = await asyncio.wait_for(
                graph.ainvoke(initial_state, config=full_config),
                timeout=QUERY_GRAPH_TIMEOUT_SEC,
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
        _trace_store[session_id] = [step.model_dump() if hasattr(step, 'model_dump') else dict(step) for step in trace_steps]
        await _persist_trace(session_id, _trace_store[session_id])

        def serialize_model(obj):
            if obj is None:
                return None
            if hasattr(obj, 'model_dump'):
                return obj.model_dump()
            if isinstance(obj, dict):
                return obj
            return str(obj)

        parallel_timing = final_state.get("parallel_timing")

        ragas = final_state.get("ragas_scores")
        eval_mode = (
            ragas.evaluation_mode
            if ragas and hasattr(ragas, "evaluation_mode")
            else "unknown"
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
            correction_history=[serialize_model(c) for c in final_state.get("correction_history", [])],
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

    except Exception as e:
        error_trace = [{
            "node_name": current_node or "unknown",
            "status": "error",
            "summary": str(e),
            "detail": {"exception": type(e).__name__}
        }]
        _trace_store[session_id] = error_trace
        await _persist_trace(session_id, error_trace)

        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "node": current_node,
                "session_id": session_id
            }
        )


ACCEPTED_EXTENSIONS = {'.pdf', '.txt', '.md'}


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
            detail=f"Unsupported file type: {detected_type}. Allowed: PDF, TXT, Markdown."
        )

    if _ingest_semaphore.locked():
        raise HTTPException(
            status_code=429,
            detail="Too many concurrent ingestion requests. Please try again shortly."
        )
    await _ingest_semaphore.acquire()
    try:
        chunker = DocumentChunker()

        if ext == '.pdf':
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                pages = chunker.load_pdf(tmp_path)
                chunks = chunker.chunk(pages, source=filename)
            finally:
                os.unlink(tmp_path)
        else:
            text = content.decode('utf-8', errors='ignore')
            pages = chunker.load_text(text, filename)
            chunks = chunker.chunk(pages, source=filename)

        indexer = get_dual_indexer()
        result = await indexer.index_chunks(chunks)

        doc = {
            "filename": filename,
            "chunk_count": len(chunks),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "status": result.get("vector", "unknown"),
        }
        _ingested_docs.append(doc)
        await _persist_ingested_doc(filename, len(chunks), len(content))

        return IngestResponse(
            filename=filename,
            chunk_count=len(chunks),
            status="indexed",
            mode=result.get("mode"),
            bm25=result.get("bm25"),
            vector=result.get("vector"),
            chunks=[{
                "chunk_id": c["chunk_id"],
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "token_count": c["token_count"],
                "preview": c["content"][:100] + "..." if len(c["content"]) > 100 else c["content"]
            } for c in chunks[:5]],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "filename": filename
            }
        )
    finally:
        _ingest_semaphore.release()


@api_router.get("/trace/{session_id}", response_model=TraceResponse)
async def get_trace(session_id: str):
    trace_steps = _trace_store.get(session_id, [])

    if not trace_steps:
        # Fall back to PostgreSQL
        pg_trace = await _load_trace(session_id)
        if pg_trace:
            trace_steps = pg_trace
            _trace_store[session_id] = trace_steps  # repopulate cache

    if not trace_steps:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"No trace found for session {session_id}",
                "session_id": session_id
            }
        )

    return TraceResponse(
        session_id=session_id,
        trace_steps=trace_steps
    )


@api_router.get("/stats")
async def get_stats():
    pg_connected = await vector_store.is_connected()
    cache_stats = await semantic_cache.stats()
    return {
        "indexed_documents": len(_ingested_docs),
        "bm25_doc_count": bm25_index.count(),
        "vector_doc_count": await vector_store.count() if pg_connected else 0,
        "cache_entries": cache_stats["total_entries"],
        "cache_hits": cache_stats["total_hits"],
        "total_queries_processed": len(_trace_store),
        "stub_mode": _compute_stub_mode(),
    }


async def _run_eval_background(job_id: str) -> None:
    """Execute the benchmark in-process; updates _eval_jobs for polling."""
    from axiom.eval_suite.runner import EvalRunner
    from axiom.eval_suite.benchmark import BENCHMARK_QUERIES

    runner = EvalRunner()
    suite_start = time.perf_counter()
    results: List[Dict[str, Any]] = []

    try:
        await runner._ensure_services()
        _eval_jobs[job_id]["total"] = len(BENCHMARK_QUERIES)

        for i, bq in enumerate(BENCHMARK_QUERIES):
            session_id = f"eval-{job_id}-{i:02d}"
            res = await runner.run_single(bq, session_id)
            results.append(res)
            _eval_jobs[job_id]["progress"] = i + 1
            _eval_jobs[job_id]["latest"] = res
            _eval_jobs[job_id]["results"] = list(results)

        total_s = time.perf_counter() - suite_start
        runner.results = results
        aggregate = runner._compute_aggregate(total_s, results)
        runner.save_results(aggregate)

        _eval_jobs[job_id]["status"] = "complete"
        _eval_jobs[job_id]["aggregate"] = aggregate
        _eval_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        await _persist_eval_run(job_id, _eval_jobs[job_id])
    except Exception as exc:
        logger.exception("Eval job %s failed: %s", job_id, exc)
        _eval_jobs[job_id]["status"] = "failed"
        _eval_jobs[job_id]["error"] = str(exc)
        _eval_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        await _persist_eval_run(job_id, _eval_jobs[job_id])


async def _run_eval_with_semaphore(job_id: str) -> None:
    async with _eval_semaphore:
        await _run_eval_background(job_id)


@api_router.post("/eval/run", dependencies=[Depends(require_api_key)])
@limiter.limit("2/hour")
async def run_eval_suite(request: Request, background_tasks: BackgroundTasks):
    """Start the 30-query benchmark in the background. Poll GET /api/eval/status/{job_id}."""
    if _eval_semaphore.locked():
        raise HTTPException(
            status_code=409,
            detail="An evaluation run is already in progress."
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
    background_tasks.add_task(_run_eval_with_semaphore, job_id)

    return {
        "job_id": job_id,
        "status": "started",
        "poll_url": f"/api/eval/status/{job_id}",
        "message": "Poll poll_url until status is complete; results are written to eval_results.json on success.",
    }


@api_router.get("/eval/status/{job_id}")
async def eval_job_status(job_id: str):
    """Live progress for a benchmark job started via POST /api/eval/run."""
    job = _eval_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "Job not found", "job_id": job_id})
    return {"job_id": job_id, **job}


@api_router.post("/eval/run/stream", dependencies=[Depends(require_api_key)])
async def run_eval_suite_stream():
    """Stream SSE progress after each query (optional; use curl -N). Saves results on success."""
    from axiom.eval_suite.runner import EvalRunner
    from axiom.eval_suite.benchmark import BENCHMARK_QUERIES

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
        except Exception as exc:
            logger.exception("Eval stream failed: %s", exc)
            yield f"data: {json.dumps({'final': True, 'error': str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api_router.get("/eval/results")
async def get_eval_results():
    """Return the last saved eval_results.json if it exists."""
    import json
    results_path = Path(__file__).parent / "eval_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="No eval results found. Run POST /api/eval/run first.")
    with open(results_path) as f:
        return json.load(f)


@api_router.get("/session/{session_id}/state")
async def get_session_state(session_id: str, request: Request):
    """Return the last checkpointed state for a session."""
    try:
        graph = get_graph(checkpointer=request.app.state.checkpointer)
        config = {"configurable": {"thread_id": session_id}}
        state = await graph.aget_state(config)
        if state is None or state.values is None:
            raise HTTPException(status_code=404, detail="No state found for session")
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
