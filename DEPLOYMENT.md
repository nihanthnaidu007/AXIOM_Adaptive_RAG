# AXIOM Deployment Guide

## Why Vercel Does Not Work for the Backend

The AXIOM backend is a persistent, stateful process. It cannot run on Vercel or any serverless platform. Four hard incompatibilities:

1. LangGraph uses MemorySaver which stores all pipeline state in process RAM. Serverless cold starts destroy all in-flight state.
2. The BM25 index is built at startup by loading all chunks from PostgreSQL. Serverless invocations do not share this memory.
3. The embedding cache is an in-process LRUCache. Lost on every cold start.
4. The server uses uvicorn with keep-alive of 1800 seconds. Vercel Pro maximum function timeout is 60 seconds.

The React frontend is fully Vercel-compatible. Deploy it there.
The backend must run on a persistent server.

## Recommended Production Stack

| Component  | Service                        | Reason                          |
|------------|--------------------------------|---------------------------------|
| Backend    | Railway or Render              | Persistent process, free tier   |
| PostgreSQL | Railway Postgres plugin        | Managed, pgvector supported     |
| Redis      | Railway Redis plugin           | Managed, same network as app    |
| Frontend   | Vercel                         | Static SPA, zero config         |
| Evaluation | Claude API (replace Ollama)    | Ollama cannot run in cloud      |

## Railway Deployment (Recommended)

1. Push the repo to GitHub.
2. Create a new Railway project.
3. Add a Railway Postgres plugin — copy the DATABASE_URL it provides.
4. Add a Railway Redis plugin — copy the REDIS_URL it provides.
5. Add a new Railway service pointing to the repo.
6. Set the root directory to backend/.
7. Set the start command to:
   uvicorn server:app --host 0.0.0.0 --port $PORT
8. Set all environment variables from .env.example in the Railway Variables panel. Use the DATABASE_URL and REDIS_URL from the plugins.
9. Deploy. Railway will run pip install -r requirements.txt automatically.

## Frontend Deployment (Vercel)

1. Push the repo to GitHub.
2. Create a new Vercel project pointing to the frontend/ directory.
3. Set build command: npm run build
4. Set output directory: build
5. Add environment variable:
   REACT_APP_BACKEND_URL=https://your-railway-backend-url.up.railway.app
6. Deploy.

## Environment Variables Required in Production

Copy .env.example and fill in every value before deploying.
Do not use the placeholder values. Generate strong random strings for API_KEY, POSTGRES_PASSWORD, and REDIS_PASSWORD.

## Evaluation

AXIOM uses Claude Haiku (claude-haiku-4-5-20251001) for RAGAS evaluation by default.
This works in every deployment environment. No Ollama installation is required.

To use local Ollama instead, set USE_CLAUDE_EVALUATOR=false in your environment.
Ollama must be running and reachable at OLLAMA_HOST for this to produce real scores.
If Ollama is selected but unreachable, evaluation returns parse_error results
(treated as below-threshold, not mock passes).
