# AXIOM_Adaptive_RAG — Codebase Map

Folder-level overview (depth 2). Backend is a Python/FastAPI + LangGraph service; frontend is a CRA/React dashboard. Read `README.md` first — it documents the 13-node pipeline, routing logic, and every env var in detail.

| Path | Kind | Purpose |
|---|---|---|
| `backend/` | Python app | FastAPI service hosting the RAG pipeline |
| `backend/server.py` | entrypoint | FastAPI app: lifespan wiring, `/api/query`, `/api/query/stream` (SSE), `/api/ingest`, `/api/health`, `/api/stats`, eval-suite endpoints, rate limiting, X-API-Key gate, persistence tables |
| `backend/axiom/` | package | Core domain code (everything except the HTTP layer) |
| `backend/axiom/graph/` | code | LangGraph StateGraph: `builder.py`/`graph.py` composition, `state.py` TypedDict, `sub_query_runner.py`, `nodes/` = 13 pipeline nodes (classify, cache, route, retrieve_bm25/vector/hybrid, decompose, rerank, web_search, generate, evaluate, rewrite, finalize) |
| `backend/axiom/retrieval/` | code | `bm25_index.py` (in-memory, hydrated from pg on boot), `vector_store.py` (async SQLAlchemy + pgvector), `embeddings.py` (OpenAI singleton), `hybrid_fusion.py` (RRF k=60), `reranker.py` (cross-encoder) |
| `backend/axiom/evaluation/` | code | RAGAS scoring: `claude_evaluator.py` (default Haiku), `critic_llm.py` (Ollama), `ragas_scorer.py`, `thresholds.py` (confidence bands) |
| `backend/axiom/cache/` | code | `semantic_cache.py` — Redis two-tier (exact hash + cosine over recent 200) |
| `backend/axiom/ingest/` | code | `loader.py` (pdfplumber/tiktoken/NLTK chunker), `indexer.py` (dual BM25+pgvector writer) |
| `backend/axiom/search/` | code | `web_search.py` — Tavily client wrapper |
| `backend/axiom/observability/` | code | `langsmith.py` — tracing config |
| `backend/axiom/eval_suite/` | code | `benchmark.py` (30 queries/6 categories), `runner.py`, `stress_test.py` |
| `backend/axiom/config.py` | code | `AxiomConfig` (pydantic-settings) — every threshold/top_k/timeout; loads root `.env` |
| `backend/axiom/llm/` | code | Shared Anthropic client singleton |
| `backend/alembic/` | migrations | Alembic env + `versions/22496c2e6b17_initial_schema.py`; server also auto-creates tables at startup |
| `backend/tests/` | tests | 60-test pytest suite (offline, network stubbed) — CI runs this |
| `backend/scripts/` | ops | `start.sh` — docker-compose + uvicorn launcher |
| `frontend/` | JS app | React 18 dashboard (CRA + craco, Tailwind, MERIDIAN design system) |
| `frontend/src/` | code | `App.js` (dashboard shell + SSE consumer), `config.js` (backend URL), `index.css` (design tokens) |
| `frontend/src/components/axiom/` | code | `QueryInput`, `UploadPanel`, `PipelineStrip` (13-node signal trace), `SignalPanel`, `EvaluationPanel`, `CorrectionRecord`, `AnswerPanel`, `StatusBar`, `HexBackground` |
| `frontend/src/components/ui/` | code | Radix-based shared UI primitives |
| `frontend/public/` | static | CRA static assets |
| `.github/workflows/` | CI | `ci.yml` — backend pytest (PG+Redis service containers, dummy LLM keys), frontend `npm ci` + build + no-source-maps check |
| `.github/ISSUE_TEMPLATE/` | meta | Bug/feature issue templates |
| `Images/` | docs | README screenshots (dashboard idle → final answer) |
| root | docs/config | `README.md`, `CONTRIBUTING.md`, `DEPLOYMENT.md`, `SECURITY.md`, `docker-compose.yml` (pgvector:pg16 + redis:7), `.env.example`, `LICENSE` (MIT) |

**Where to make common changes:** pipeline behavior → `backend/axiom/graph/nodes/` + `graph.py`; retrieval tuning → `backend/axiom/retrieval/` + `config.py`; API surface → `backend/server.py` (and update README's endpoint table); UI panels → `frontend/src/components/axiom/`; thresholds/env → `axiom/config.py` + `.env.example` + README reference table (repo convention: update both).
