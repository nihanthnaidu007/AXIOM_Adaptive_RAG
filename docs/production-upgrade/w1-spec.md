> Source: Obvious artifact `art_0TKxEV4g` — “AXIOM W1 Grounded Spec — tests, CI, schema, envelopes” · exported 2026-09-17

# W1 Spec — AXIOM_Adaptive_RAG: Integration tests, CI gates, schema single-source, error envelopes

Branch: `feat/w1-tests-ci-schema` (from origin/main @ da94af5). Repo: nihanthnaidu007/AXIOM_Adaptive_RAG.

## Grounding — verified this session (file reads, not memory)

- `backend/alembic/env.py` — runs `engine_from_config` (sync) over `DATABASE_URL` defaulting to `postgresql+asyncpg://…` → MissingGreenlet crash. Requirements pin `asyncpg` only; **no psycopg** in `backend/requirements.txt` → fix must be the async engine + `run_sync` pattern, not a driver swap.
- `backend/server.py` — `_create_persistence_tables()` runs three ad-hoc `CREATE TABLE IF NOT EXISTS` blocks (pipeline_traces, ingested_documents, eval_runs) at startup, duplicating migration `22496c2e6b17`. Alembic is never invoked at startup.
- `backend/axiom/retrieval/vector_store.py` — `connect()` also self-creates `chunk_embeddings` + ivfflat index; the migration has the table but **not the index**.
- Raw-exception leak sites in `server.py` (6): `/query` 500 (`str(e)`), `/query/stream` SSE error event (`str(exc)`), `/ingest` 500, `DELETE /documents` 500 (`f"Delete failed: {exc}"`), `/eval/run/stream` SSE (`str(exc)`), `/session/{id}/state` 500.
- CI (`.github/workflows/ci.yml`) — two jobs (backend tests w/ pgvector+redis services, frontend build). No ruff/mypy/eslint, no migration verification.
- `frontend/` — no ESLint config file; ESLint 9 + plugins already in devDependencies. `UploadPanel.js:72` reads `err.detail?.error || err.detail` → the envelope must keep `detail.error` a **string**.
- W0 test suite asserts `r.json()["detail"] == {"error": "Invalid API key"}` and `detail["error"]` string guidance → intentional error shapes stay frozen; only leak sites change.
- Sandbox services: PG 17 with pgvector available; redis on 6380 (6379 is occupied by a password-protected instance).

## Design

### 1. Alembic as schema single-source
- `env.py`: async engine (`async_engine_from_config`) + `connection.run_sync(do_run_migrations)`; sync-URL URLs keep the plain engine. No new dependency.
- New migration `0002`: add `chunk_embeddings_vec_idx` ivfflat index `IF NOT EXISTS` so a migration-built schema matches runtime expectations.
- Startup: `lifespan` runs `alembic upgrade head` in a worker thread (`asyncio.to_thread`) **before** `vector_store.connect()`, gated by new config `run_migrations_on_startup` (env `RUN_MIGRATIONS_ON_STARTUP`, default true). Failures log a warning and continue (graceful degradation, consistent with W0 behavior).
- `_create_persistence_tables()` deleted — tables come from migration `22496c2e6b17`.
- `vector_store.connect()` keeps its idempotent DDL as documented fallback (no-op once migrations ran; needed by the ASGITransport test harness, which does not run lifespan). Fresh-DB truth = `alembic upgrade head` (CI-verified).

### 2. Structured error envelopes (`backend/axiom/api_errors.py`)
- `error_detail(code, message, **context) -> dict` — returns `{"error": <safe message str>, "code": <machine code>, "context": {...}}`.
- HTTP failure paths raise `HTTPException(500, detail=error_detail("internal_error", "Internal server error...", session_id=…))` — `detail.error` stays a string (frontend-safe), full exception goes to server logs via `logger.exception`.
- SSE error events: `{"type": "error", "code": ..., "message": <safe>}`.
- Applied to all six leak sites.

### 3. Integration tests
- `tests/test_api_integration.py` — fake graph monkeypatched at `server.get_graph` (+ stubbed `app.state.checkpointer`):
  - `/query`: happy 200 full-shape; 400 empty/too-long/bad-uuid; 500 envelope (no raw exception text); 504 timeout path.
  - `/query/stream`: SSE frames (node_complete → done → [DONE]); astream_events-failure fallback; error event envelope; 400 validation.
  - `/trace`: populated after a fake /query; 404 shape for unknown.
  - `/session/{id}/state`: 200 via fake `aget_state`; 404 unknown.
  - `/eval/run` started (background task stubbed), `/eval/results` 404 shape, `/eval/run/stream` progress + error-envelope events.
  - middleware 413; ingest validation 400/413/415 (no services needed); `/health` public.
- `tests/test_migrations.py` — creates a scratch database, runs `alembic upgrade head` via subprocess with overridden `DATABASE_URL`, asserts all tables + ivfflat index + vector extension exist via information_schema, then `downgrade base`. Skips when PG absent.
- `tests/conftest.py` — shared `client` + limiter-reset fixtures move here (W0 file slimmed to use them).

### 4. CI (`.github/workflows/ci.yml`)
- `backend-lint` — ruff check.
- `backend-typecheck` — mypy (pragmatic config).
- `frontend-lint` — `npm run lint` (new `lint` script + `eslint.config.js`).
- `migrations-fresh-db` — empty PG service → `alembic upgrade head` → run `tests/test_migrations.py` schema assertions.
- Existing `backend-tests` and `frontend-build` unchanged.

### 5. Lint/type configs
- `backend/pyproject.toml`: `[tool.ruff]` (default rule set, line-length 100, excludes alembic) + `[tool.mypy]` (ignore_missing_imports, exclude alembic/tests; per-module relaxation only for pre-existing noise, new code clean).
- `frontend/eslint.config.js`: flat config on @eslint/js recommended + react/react-hooks/jsx-a11y; violations fixed or pruned with named rationale.

## Verification
1. Clean local DB → `alembic upgrade head` → schema assertions green; downgrade round-trip.
2. Full pytest locally against PG 5432 + Redis 6380.
3. ruff, mypy, eslint, npm build all clean locally.
4. CI green on the PR (all six jobs).

## Out of scope (flagged, not fixed)
- Embedding-dimension migration path (shortlist 8), single-process store redesign (4), Prometheus/metrics (10).

## Load-bearing assumption
"Migrations-only schema" means: a fresh DB reaches full schema via `alembic upgrade head` alone, and startup no longer hand-creates tables — verified by the fresh-DB CI job and migration tests. Custom `embedding_dimensions` deployments remain a known limitation (shortlist 8), unchanged by this wave.
