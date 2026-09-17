> Source: Obvious artifact `art_mEKQ5bhY` — “AXIOM_Adaptive_RAG Production-Readiness Survey” · exported 2026-09-17

# AXIOM_Adaptive_RAG Production-Readiness Survey

*Read-only survey compiled from the coder worker's report (thread th_nwTrQYTA). Findings cite concrete files in nihanthnaidu007/AXIOM_Adaptive_RAG. No files were modified.*

## 1. What the product is

Self-hosted adaptive RAG platform (AXIOM Intelligence Platform v1.5.0, `backend/pyproject.toml`): FastAPI backend (`backend/server.py`, ~1,186 lines) + React CRA frontend. LangGraph StateGraph pipeline (`backend/axiom/graph/graph.py`): classify_query → check_cache (Redis semantic cache) → route_retrieval (BM25/vector/hybrid via RRF, `backend/axiom/retrieval/hybrid_fusion.py`) → decompose_query → rerank (sentence-transformers cross-encoder) → generate (Anthropic Claude, `backend/axiom/llm/client.py`) → evaluate (RAGAS-style via Claude Haiku or local Ollama) → self-correction loop (max 3 rewrites) → Tavily web-search fallback. Ingestion: pdfplumber + NLTK/tiktoken chunking, dual-index writer to in-memory BM25 + pgvector. Endpoints: `/api/query`, `/api/query/stream` (SSE), `/api/ingest`, `/api/trace/{id}`, `/api/stats`, `/api/health`, `/api/eval/*`, `/api/session/{id}/state`. Postgres (pgvector, ivfflat) + Redis via docker-compose; Railway backend + Vercel frontend per DEPLOYMENT.md. Target user: single-operator developer deploying their own document-QA instance.

## 2. Maturity signals

| Signal | State |
| --- | --- |
| Tests | 8 unit test files (~740 lines) with stubbed network; **no API/integration tests** — the HTTP surface is untested |
| CI | Real two-job workflow (backend pytest on pgvector/redis services + frontend build with source-map check); latest main run green (35243754249); no lint/typecheck/coverage jobs |
| Error handling | Good — typed retries with jitter on Anthropic transient errors, graceful startup degradation, graph timeouts |
| Secrets | Clean scan — no hardcoded keys; pydantic-settings config with required-key validator |
| Deployment | docker-compose provides postgres+redis only; **no backend Dockerfile**; single uvicorn process assumed |
| Scale | In-memory trace store, ingest registry, eval jobs, and RAM-hydrated BM25 index — **breaks under >1 worker**; slowapi limiter per-process |
| Docs | Thorough 42KB README + SECURITY.md, CONTRIBUTING.md, DEPLOYMENT.md |
| Deps | requirements.txt exact-pinned; frontend on deprecated react-scripts 5/CRA + craco |

## 3. Production-readiness gaps

- **FAIL-OPEN AUTH (blocking):** `require_api_key` returns None when API_KEY is unset — all query/ingest endpoints silently unauthenticated by default. (`server.py`)


- **UNAUTHENTICATED ENDPOINTS (blocking):** `GET /api/trace/{session_id}`, `/api/session/{id}/state`, `/api/stats`, `/api/eval/*` have no auth dependency — full pipeline traces (user queries + document content) publicly readable.


- **NO DOCUMENT DELETE / CACHE INVALIDATION (blocking):** `chunk_id = sha256(source:chunk_index)` (`ingest/loader.py`) — re-uploading an updated file with the same name hits `ON CONFLICT DO NOTHING` (`vector_store.insert_chunks`), silently keeping stale chunks; no delete endpoint; Redis cache never invalidated on ingest.


- **Schema dual-source:** ad-hoc `CREATE TABLE IF NOT EXISTS` in server.py vs alembic migration `22496c2e6b17` — drift risk; alembic never wired into startup.


- **Embedding dimension drift:** ivfflat built at startup but migration hardcodes `vector(1536)` while `config.embedding_dimensions` is user-configurable — no dimension-migration path; per-process LRU embedding cache.


- **Single-process architecture:** BM25 in RAM, module-dict eval jobs/trace store — horizontal scaling impossible without redesign.


- **Observability:** optional LangSmith only; no Prometheus metrics, no request-ID propagation; raw `str(exc)` leaked to clients in SSE stream.


- **No multi-tenancy:** single global corpus; sessions UUID-guessable for trace reads.


- **Eval suite:** 30-query benchmark requires live keys, runs in-process; `/eval/run/stream` bypasses the eval semaphore.


- **Ingestion robustness:** scans-only PDFs produce zero chunks silently; no OCR fallback; no per-doc lineage (doc_id) for deletion.



## 4. Prioritized upgrade shortlist

1. **BLOCKING** — Make auth fail-closed; protect /trace, /session, /stats, /eval (`server.py`).


2. **BLOCKING** — Document delete + re-ingest versioning (content-hash chunk ids, cache invalidation on ingest).


3. Wire alembic into startup; consolidate ad-hoc CREATE TABLE blocks.


4. Replace process-local state (trace store, eval jobs, BM25, slowapi) with shared backends for >1 worker.


5. API integration tests (TestClient over /query, /ingest, /trace) in CI.


6. Lint/typecheck job (ruff + mypy) + frontend ESLint; CRA → Vite.


7. Structured error envelopes; stop echoing raw exception strings.


8. Embedding-dimension guardrails at startup + documented migration path.


9. Move eval-suite semaphore under /eval/run/stream; persist eval jobs to the eval_runs table.


10. Observability baseline: structured JSON logging, request IDs, Prometheus /metrics, per-query token-cost accounting.


11. Backend Dockerfile + full-stack compose profile for one-command self-host.


12. Frontend: token-based auth UI, upload progress/resumability, error toasts tied to API error envelopes.



*CI verified green on latest main (run 35243754249). Survey read-only; no files modified.*