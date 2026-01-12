#!/usr/bin/env bash
set -euo pipefail

echo "=== AXIOM Startup Check ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "Starting Docker services (postgres, redis)..."
docker compose up -d

echo "Waiting for Postgres (pgvector)..."
for _ in $(seq 1 30); do
  if docker compose exec -T postgres pg_isready -U axiom -d axiom_rag >/dev/null 2>&1; then
    echo "Postgres ready"
    break
  fi
  sleep 1
done

echo "Waiting for Redis..."
for _ in $(seq 1 30); do
  if docker compose exec -T redis redis-cli ping >/dev/null 2>&1; then
    echo "Redis ready"
    break
  fi
  sleep 1
done

BACKEND_DIR="$REPO_ROOT/backend"
cd "$BACKEND_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
fi

echo "Starting backend server on http://127.0.0.1:8000 ..."
exec uvicorn server:app --host 127.0.0.1 --port 8000 --timeout-keep-alive 1800
