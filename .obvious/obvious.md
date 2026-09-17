# AXIOM_Adaptive_RAG — Agent Guide

**Repo:** nihanthnaidu007/AXIOM_Adaptive_RAG (default branch: `main`)
**One-liner:** Adaptive RAG pipeline with a self-correcting hallucination detection loop — a 13-node LangGraph cyclic graph behind a FastAPI SSE API, with a React 18 dashboard.

## Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph 1.2 cyclic StateGraph, AsyncPostgresSaver checkpointer (MemorySaver fallback) |
| Generation LLM | Claude Sonnet via Anthropic API (`langchain-anthropic`) |
| Evaluation LLM | Claude Haiku 4.5 RAGAS scorer (default) or local Ollama llama3.2 (`USE_CLAUDE_EVALUATOR=false`) |
| Retrieval | BM25 (rank_bm25, in-memory), pgvector (OpenAI `text-embedding-3-small`), RRF hybrid fusion (k=60) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers (CPU) |
| Web fallback | Tavily Search (fires on zero-chunk retrieval or exhausted corrections) |
| Cache | Redis two-tier semantic cache (exact + cosine, TTL 7 days) |
| Database | PostgreSQL + pgvector extension; Alembic migrations in `backend/alembic/` |
| Backend API | FastAPI + uvicorn, SSE via StreamingResponse, slowapi rate limiting (30 req/min) |
| Frontend | React 18 + Tailwind (CRA/craco), native fetch + ReadableStream SSE consumer |
| Observability | LangSmith tracing (optional) |
| CI | GitHub Actions: backend pytest + frontend build + no-source-maps check |

## Commands

```bash
# Infrastructure (compose spec: pgvector/pgvector:pg16 + redis:7; see local-dev skill for Docker-less setup)
docker compose up -d

# Backend (Python >=3.11; repo pins in backend/requirements.txt)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab', quiet=True)"   # required for ingest chunking
uvicorn server:app --host 127.0.0.1 --port 8000 --reload          # API on :8000

# Tests (offline, network stubbed; mirror of CI)
cd backend && pytest tests/ -v

# Frontend (Node 18+; npm canonical — CI runs npm ci)
cd frontend
npm ci
npm start                                                          # dev server on :3000
npm run build                                                      # production build, must emit 0 *.map files

# Health / smoke
curl -s http://localhost:8000/api/health | python3 -m json.tool
curl -s -X POST http://localhost:8000/api/ingest -F "file=@doc.txt;type=text/plain"
curl -s -N -X POST http://localhost:8000/api/query/stream -H "Content-Type: application/json" -d '{"query":"...","session_id":null}'
```

`backend/scripts/start.sh` orchestrates docker-compose + uvicorn in one shot.

## Codebase Map

Full folder map: [codebase-map.md](codebase-map.md). Shape in one paragraph: `backend/` (FastAPI app `server.py` + `axiom/` package: `graph/` 13 nodes, `retrieval/` BM25+pgvector+RRF+reranker, `evaluation/` RAGAS, `cache/` Redis, `ingest/` chunker+indexer, `search/` Tavily, `eval_suite/` benchmark; `alembic/` migrations; `tests/` 60-test pytest suite) and `frontend/` (CRA app: `src/App.js` dashboard + `src/components/axiom/` UI panels). Root: `docker-compose.yml`, `.env.example`, `README.md` (excellent, read it first), `CONTRIBUTING.md`, `DEPLOYMENT.md`, `SECURITY.md`.

## Environment

Copy `.env.example` → `.env` at repo **root** (server.py loads from there). Required non-empty: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `POSTGRES_URL` or `DATABASE_URL`. Optional: `TAVILY_API_KEY`, `API_KEY`, `CORS_ORIGINS`, `USE_CLAUDE_EVALUATOR`, `LANGCHAIN_*`. Full reference table in README.

Offline / CI mode: dummy values for the two LLM keys (CI uses `ci-dummy-*`) keep the stack fully bootable — tests stub network, ingest still indexes BM25, queries run the whole graph and return honest degraded states (`evaluation_mode: "parse_error"`, `stub_mode: true`). The system never fakes a passing gate.

## Local Verification

> **Validation Summary (onboarding run, 2026-09-17)**
> - Infra: PostgreSQL 17.11 + pgvector 0.8.0 and Redis 8.0.2 running natively (apt; no Docker in sandbox). DB `axiom_rag`, user `axiom`.
> - Backend: venv on Python 3.13.14, all requirements installed (CPU-only torch), NLTK `punkt_tab` downloaded. `uvicorn server:app` on **127.0.0.1:8000** — startup banner: pgvector connected, Redis connected, reranker loaded, evaluator claude-haiku/unreachable (dummy key), web_search not_configured (no Tavily key).
> - Tests: `pytest tests/ -v` → **60 passed** in 3.84s.
> - Primary flow 1 (ingest): `POST /api/ingest` TXT → chunked (168 tokens), BM25 indexed; vector insert honestly failed on dummy embedding key.
> - Primary flow 2 (query): `POST /api/query` → full 13-node pipeline end-to-end (classify→cache→route→hybrid retrieve→rerank `reranker_mode=real`→generate→evaluate→3× rewrite→finalize), 1819 ms, honest `UNRELIABLE/0.0` confidence.
> - Primary flow 3 (SSE): `POST /api/query/stream` → per-node `node_complete` events, terminates `data: [DONE]`.
> - Frontend: `npm ci` + `npm start` → webpack compiled, dev server on **:3000** serving the AXIOM dashboard; `npm run build` → success, **0** source-map files.
> - Browser (headless Chromium via Playwright): 3 screenshots (idle dashboard, query entered, post-RUN), **0 page errors, 0 console errors**; rendered UI showed live health pills (pg/redis), correction record, and the honest UNRELIABLE answer band.
> - Verdict: `dev_stack_healthy: true`.

Re-verify quickly after any resume: `curl -s localhost:8000/api/health` (expect `status: ok`, pgvector/redis `connected`) and `curl -s -o /dev/null -w '%{http_code}' localhost:3000` (expect 200).

## Sandbox Snapshot

- **snapshotId:** `ssqv4kslz8prod5g9y41:default`
- **captured:** 2026-09-17T15:56:06.995Z
- **state at capture:** full dev stack live — Postgres(5432)+pgvector, Redis(6379), uvicorn(8000, tmux `backend`), CRA dev server(3000, tmux `frontend`); 60/60 tests green; 1 test document indexed.
- Restoring this snapshot resumes the sandbox; if tmux sessions are gone, restart per [skills/local-dev/SKILL.md](skills/local-dev/SKILL.md).

## Known Issues / Notes

1. **`POSTGRES_URL` dialect bug:** `.env.example` and README document `postgresql+psycopg://…`, but `server.py` passes this value raw to LangGraph's `AsyncPostgresSaver.from_conn_string()` → psycopg raises `missing "="` and **startup fails**. Use a plain `postgresql://user:pass@host:5432/db` URI for `POSTGRES_URL`. (Keep `postgresql+asyncpg://…` for `DATABASE_URL` — that one is SQLAlchemy-side and works.)
2. Compose pins `pgvector/pgvector:pg16`; the sandbox snapshot used native PostgreSQL 17 + pgvector 0.8.0 (apt) — functionally equivalent for this codebase.
3. `/api/health` reports `checkpointing: "enabled (MemorySaver)"` as a static label; when pgvector connects, the real checkpointer is `AsyncPostgresSaver`.
4. Frontend has no unit tests by design (CI = build + no source maps). Manual browser smoke per local-dev skill.
