---
name: local-dev
description: Bring up and verify the AXIOM dev stack (Postgres+pgvector, Redis, FastAPI backend, React frontend) in this sandbox, including the Docker-less native-service path and offline dummy-key mode.
---

# Local Dev — AXIOM_Adaptive_RAG

Durable record of the onboarding run (2026-09-17). Everything below was executed and verified in this sandbox.

## Prerequisites

- Python ≥3.11 (sandbox: 3.13.14), Node 18+ (sandbox: 20.20.2), apt + sudo.
- **No Docker in this sandbox.** `docker-compose.yml` (pgvector/pgvector:pg16 + redis:7) is the canonical infra spec; when Docker exists, `docker compose up -d` replaces steps 1–2. Otherwise use the native path below — it is equivalent (PG 17 + pgvector 0.8.0 instead of PG 16).

## Bring-up (verified sequence)

1. **Native services (Docker-less path):**
   ```bash
   sudo apt-get update && sudo apt-get install -y postgresql-17 postgresql-17-pgvector redis-server libmagic1t64
   sudo pg_ctlcluster 17 main start                       # Postgres on :5432
   redis-server --daemonize yes --port 6379 --requirepass axiom_local_dev --save '' --appendonly no
   sudo -u postgres psql -c "CREATE ROLE axiom WITH LOGIN PASSWORD 'axiom_local_dev';"
   sudo -u postgres psql -c "CREATE DATABASE axiom_rag OWNER axiom;"
   sudo -u postgres psql -d axiom_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```
2. **Root `.env`** (`cp .env.example .env`, then edit). Offline/CI mode keeps the stack fully meaningful without real keys:
   - `ANTHROPIC_API_KEY=ci-dummy-anthropic-key`, `OPENAI_API_KEY=ci-dummy-openai-key` (tests stub network; server degrades honestly — see below)
   - `REDIS_PASSWORD=axiom_local_dev`, `POSTGRES_USER/PASSWORD=axiom/axiom_local_dev`, `POSTGRES_DB=axiom_rag`
   - **`POSTGRES_URL=postgresql://axiom:axiom_local_dev@localhost:5432/axiom_rag`** ← plain URI, NOT `postgresql+psycopg://…` (see Gotcha 1)
   - `DATABASE_URL=postgresql+asyncpg://axiom:axiom_local_dev@localhost:5432/axiom_rag`
   - `API_KEY=` (empty disables the X-API-Key gate), `CORS_ORIGINS=http://localhost:3000`
3. **Backend deps** (CPU torch first — the reranker is CPU-only; avoids the ~2 GB CUDA wheel):
   ```bash
   cd backend && python3 -m venv .venv
   .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
   .venv/bin/pip install -r requirements.txt
   .venv/bin/python -c "import nltk; nltk.download('punkt_tab', quiet=True)"   # ingest chunking needs it
   ```
4. **Run backend** (tmux recommended; reranker downloads ~90 MB from HF on first boot):
   ```bash
   tmux new-session -d -s backend -c backend ".venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000"
   # wait for: "Uvicorn running on http://127.0.0.1:8000"
   ```
5. **Run frontend:**
   ```bash
   cd frontend && npm ci
   tmux new-session -d -s frontend -c frontend "BROWSER=none CI=true npm start"   # :3000, "webpack compiled successfully"
   ```

## Verification checklist (all green on 2026-09-17 run)

```bash
cd backend && .venv/bin/python -m pytest tests/ -v          # 60 passed (~4s, offline)
curl -s localhost:8000/api/health                            # status ok; pgvector+redis connected, reranker loaded
curl -s -X POST localhost:8000/api/ingest -F "file=@/tmp/doc.txt;type=text/plain"   # BM25 indexed
curl -s -X POST localhost:8000/api/query -H "Content-Type: application/json" -d '{"query":"...","session_id":null}'
curl -s -N -X POST localhost:8000/api/query/stream -H "Content-Type: application/json" -d '{"query":"...","session_id":null}' | tail -c 200   # ends with data: [DONE]
cd frontend && npm run build && find build/static/js -name "*.map" | wc -l        # 0
```

Browser smoke (optional, headless): Playwright + `playwright install --with-deps chromium`; screenshot `http://localhost:3000`, expect 0 console/page errors and status-bar pills `pg:`/`redis:`.

## Offline dummy-key mode — expected honest degradations

With dummy LLM keys the stack boots and runs the **entire** graph, but: `evaluator: claude-haiku/unreachable`, `web_search: not_configured` (no Tavily key), `stub_mode: true`, ingest reports `vector: failed` (embedding 401) while BM25 still indexes, queries return `evaluation_mode: "parse_error"` with `UNRELIABLE/0.0` confidence and a fallback answer after 3 correction attempts. This is by design — the gate never fakes a pass. Real keys restore full behavior with zero code changes.

## Gotchas

1. **`POSTGRES_URL` must be a plain `postgresql://` URI.** The `.env.example`/README format `postgresql+psycopg://…` crashes startup inside `AsyncPostgresSaver.from_conn_string()` (psycopg rejects the dialect suffix: `missing "=" …`). `DATABASE_URL` keeps the `+asyncpg` dialect (SQLAlchemy side, works).
2. Ports: backend **8000**, frontend **3000**, Postgres **5432**, Redis **6379** — all parsed from startup output / compose spec, don't assume others.
3. `pg_ctlcluster` + manual `redis-server` are needed because the sandbox has no systemd; services do not survive sandbox rebuild — snapshot `ssqv4kslz8prod5g9y41:default` (2026-09-17T15:56:06.995Z) has everything warm.
4. CI env parity for pytest: export `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` dummies + `POSTGRES_URL`/`DATABASE_URL`/`REDIS_PASSWORD` matching local services (tests stub network, so values only matter for config validation).
5. First uvicorn boot downloads the cross-encoder from Hugging Face (~20 s); subsequent boots are instant (HF cache).
