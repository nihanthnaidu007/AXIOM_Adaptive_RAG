#!/usr/bin/env bash
# Two-worker smoke test — the W2 acceptance proof that the single-worker
# limitation is gone.
#
# Runs TWO separate uvicorn worker processes (default ports 8901/8902)
# against ONE Postgres + ONE Redis, writes a trace through worker A, and
# reads it back through worker B. Before W2, eval-job state lived in a
# per-process dict and traces were only written through on the same worker;
# now Postgres is the source of truth.
#
# Usage:
#   backend/scripts/two_worker_smoke.sh
# Env:
#   POSTGRES_URL / DATABASE_URL, REDIS_HOST/PORT/PASSWORD, API_KEY
#   (defaults come from the repo-root .env via server.py), plus
#   WORKER_A_PORT, WORKER_B_PORT to override the ports.
#
# The query POSTed to worker A is expected to fail (this environment has no
# real LLM keys) — that is fine: the failure path persists an error trace
# too. The assertion is that worker B serves that trace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BACKEND_DIR"

PYTHON="${PYTHON:-.venv/bin/python}"
PORT_A="${WORKER_A_PORT:-8901}"
PORT_B="${WORKER_B_PORT:-8902}"

# Load the repo-root .env (same file server.py loads) so the Postgres ping,
# the exported API_KEY default, and both workers resolve identical settings.
# load_dotenv(override=False) in server.py keeps these exported values
# authoritative for the workers as well.
if [[ -f "$BACKEND_DIR/../.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$BACKEND_DIR/../.env"
  set +a
fi

API_KEY="${API_KEY:-smoke-test-key}"
export API_KEY

if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3)"; fi

echo "== Two-worker smoke test =="
echo "worker A: http://127.0.0.1:${PORT_A}  worker B: http://127.0.0.1:${PORT_B}"

# Fail fast when Postgres is unreachable — the whole point is shared state.
"$PYTHON" - <<'PY'
import asyncio, os, sys
import asyncpg

async def main():
    url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL", "")
    url = url.replace("postgresql+asyncpg://", "postgresql://").replace("postgresql+psycopg://", "postgresql://")
    conn = await asyncpg.connect(url)
    await conn.close()
    print("Postgres reachable")

asyncio.run(main())
PY

"$PYTHON" -m uvicorn server:app --host 127.0.0.1 --port "$PORT_A" --workers 1 \
  > /tmp/axiom_smoke_worker_a.log 2>&1 &
PID_A=$!
"$PYTHON" -m uvicorn server:app --host 127.0.0.1 --port "$PORT_B" --workers 1 \
  > /tmp/axiom_smoke_worker_b.log 2>&1 &
PID_B=$!
trap 'kill "$PID_A" "$PID_B" 2>/dev/null || true' EXIT

wait_for_health() {
  local port="$1"
  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${port}/api/health" > /dev/null 2>&1; then
      echo "worker on :${port} healthy"
      return 0
    fi
    sleep 1
  done
  echo "worker on :${port} failed to become healthy" >&2
  tail -20 "/tmp/axiom_smoke_worker_${2}.log" >&2 || true
  return 1
}
wait_for_health "$PORT_A" a
wait_for_health "$PORT_B" b

SESSION_ID="$("$PYTHON" -c "import uuid; print(uuid.uuid4())")"
echo "session: $SESSION_ID"

echo "-- POST /api/query on worker A --"
QUERY_STATUS=$(curl -s -o /tmp/axiom_smoke_query.json -w "%{http_code}" \
  -X POST "http://127.0.0.1:${PORT_A}/api/query" \
  -H "X-API-Key: ${API_KEY}" -H "Content-Type: application/json" \
  -d "{\"query\": \"two-worker smoke test\", \"session_id\": \"${SESSION_ID}\"}")
echo "worker A query status: ${QUERY_STATUS}"
# 200 (real answer) and 500 (LLM keys absent — error trace is persisted too)
# both leave a trace; anything else (401/400) means a setup problem.
if [[ "$QUERY_STATUS" != "200" && "$QUERY_STATUS" != "500" ]]; then
  echo "FAIL: unexpected query status ${QUERY_STATUS}: $(cat /tmp/axiom_smoke_query.json)" >&2
  exit 1
fi

echo "-- GET /api/trace on worker B (different process, same Postgres) --"
TRACE_STATUS=$(curl -s -o /tmp/axiom_smoke_trace.json -w "%{http_code}" \
  "http://127.0.0.1:${PORT_B}/api/trace/${SESSION_ID}" \
  -H "X-API-Key: ${API_KEY}")
echo "worker B trace status: ${TRACE_STATUS}"

"$PYTHON" - "$TRACE_STATUS" <<'PY'
import json, sys

status = int(sys.argv[1])
with open("/tmp/axiom_smoke_trace.json") as f:
    body = json.load(f)

assert status == 200, f"expected 200 from worker B, got {status}: {body}"
steps = body.get("trace_steps", [])
assert steps, f"trace_steps empty in worker B response: {body}"
assert body.get("session_id"), body
print(f"PASS: worker B served {len(steps)} trace step(s) written by worker A")
print("trace step node names:", [s.get("node_name") for s in steps])
PY

echo "PASS: trace written by worker A is visible through worker B."
