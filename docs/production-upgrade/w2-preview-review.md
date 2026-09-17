> Source: Obvious artifact `art_foQIGjIf` — “AXIOM W2 PR #4 Preview Review (ee155ea) — Findings” · exported 2026-09-17

# AXIOM W2 PR #4 — Independent Preview Review (ee155ea)

**Repo:** `nihanthnaidu007/AXIOM_Adaptive_RAG` · **PR:** [#4](https://github.com/nihanthnaidu007/AXIOM_Adaptive_RAG/pull/4) `feat(w2): observability + scale` · **Reviewed SHA:** `ee155ea` (head) vs base `b4fe841` (W1 merge) · single-commit diff, 16 files, +1432/−31.

**State at review time:** the PR is already **MERGED** (squash `0878bfa` on `main`, 2026-09-17T19:26:53Z, by `app/obvious-autobuild`). This was commissioned as a preview; treat it as the independent audit of the exact merged diff. The sandbox worker had local commits beyond `ee155ea` (`1f5d9ba`), which are outside this review.

**Method:** strictly read-only — `git archive ee155ea` exported to `/tmp/axiom-w2-review`; no working-tree edits, no pushes, no PR comments. CI verified live via `gh` for run 35264548838 pinned to `ee155ea`.

**CI at ee155ea (verified, not assumed):** 6/6 pass — Backend Tests, Backend Ruff Lint, Backend mypy, Frontend Build, Frontend ESLint, Alembic fresh-database verification.

---

## Criterion 1 — Structured JSON logging with request IDs · **MET**

- **Locked 401 envelope untouched.** `require_api_key` raises `detail={"error": "Invalid API key"}` (server.py); the enrichment handler `http_exception_with_request_id` adds a `request_id` **only** when `detail` already carries a `context` dict — the 401/400/413/429 envelopes have none and pass through byte-for-byte. The W1 lock is still asserted exactly at `backend/tests/test_auth_and_lifecycle.py:147` (`assert r.json()["detail"] == {"error": "Invalid API key"}`), and the new test `test_http_exception_envelope_shape_preserved` re-locks a context-free 400.
- **Request IDs in logs and headers.** `request_context` middleware assigns the ID (client header if safe, else `uuid4().hex`), binds it to a `ContextVar`, sets the `X-Request-ID` response header, and emits a "request completed" log record with `request_id` + method/path/status/duration extras. `JsonLogFormatter` emits one JSON object per line with the ID; 500-envelopes carry it in `context.request_id`; SSE error events carry it via the new optional `sse_error_event(request_id=...)` (the only `api_errors.py` change — additive).
- **Strict validation, no mangling.** `normalize_request_id` accepts printable ASCII ≤128 and **rejects the whole value** on any control char — the docstring explicitly avoids the `str.strip` trap (`\x1f` would otherwise be silently removed). Tests cover `\x01`, `\x1f`, and 129-char inputs, asserting replacement by a server-issued 32-hex ID, never an echo of the mangled value.
- Minor: uvicorn's own access logs are not JSON (root-handler only); the app's completion line is the correlation record. Cosmetic.

## Criterion 2 — Prometheus /metrics with per-query token accounting · **MET (one named gap)**

- **Metric names/labels match tests exactly.** Six families (`axiom_http_requests_total{method,endpoint,code}`, `axiom_query_latency_seconds`, `axiom_prompt_tokens_total`, `axiom_completion_tokens_total`, and the two `*_per_query` histograms), endpoint label = matched route template → bounded cardinality; 404s collapse into `unmatched`. Tests assert exact label strings and counter/histogram deltas.
- **Wired at the real call sites.** `LLMClient.chat` reports Anthropic usage after a successful create; `EmbeddingClient.embed_text`/`embed_batch` report OpenAI prompt tokens after successful creates. Both funnel into a request-scoped token scope opened in `_invoke_graph_with_metrics` (POST /api/query) and the stream `event_generator` (POST /api/query/stream), closed with `end_token_scope` even on failure. No credentials, session IDs, or query text appear in any label.
- **Gap — the Claude evaluator is not wired.** `claude_evaluator.py` builds its own `AsyncAnthropic` client (line 67) and never calls `record_llm_usage`, so evaluation tokens (Claude Haiku, the compose-default evaluator, running inside the query graph) are invisible to per-query cost metrics. The `metrics.py` docstring promises only "generation LLM calls and embedding calls," so this reads as an untracked oversight rather than a decision. Consequence: `/api/query` token metrics undercount true per-query spend — potentially by a large factor in evaluation-heavy deployments. Not a correctness or security issue; a completeness gap worth a follow-up PR.
- **Worker-count story is coherent and documented.** Prometheus default registry is per-process; the Dockerfile pins `--workers 1` per container with scale-by-replicas stated in both the Dockerfile header and DEPLOYMENT.md ("never `--workers N`"), so numbers are correct under the documented deployment. It is not *enforced* — nothing breaks if an operator raises `--workers`, they just get per-process (silently wrong) scrapes; no `PROMETHEUS_MULTIPROC_DIR` escape hatch is provided.

## Criterion 3 — Postgres-backed trace store, Alembic-only schema · **MET**

- **Migration is the sole source for its changes.** `c4d8e2f6a9b1` (revises `e7f3a9c1d2b4`, chain: `22496c2e6b17_initial_schema` → ivfflat index → this) adds `eval_runs.error TEXT` and `eval_runs.latest JSONB` with idempotent DDL and a real `downgrade()` (`DROP COLUMN IF EXISTS` both, in reverse order). Startup runs `upgrade_to_head()` (pure Alembic `command.upgrade`) behind `RUN_MIGRATIONS_ON_STARTUP`, no `create_all` anywhere in app paths.
- **Caveat, pre-existing:** `vector_store.connect()` still self-creates `chunk_embeddings` + IVFFlat index (`vector_store.py:31–46`) inside the startup lifespan. This is **inherited W1 behavior, untouched by PR #4** (the file is not in the diff), and the migration chain mirrors the same table. It slightly dilutes "Alembic-only," but it is not a regression of this PR and the test fixture documents the asymmetry honestly.
- **Fresh-database CI check covers it.** `ci.yml:154–195` job "Alembic fresh-database verification" runs `tests/test_migrations.py` against a scratch DB: upgrade-to-head on empty, **downgrade to base**, and residue assertions — the new migration's up/down are exercised by the chain test automatically.

## Criterion 4 — Backend Dockerfile + compose backend service · **MET**

- **Dockerfile:** non-root (`useradd axiom` + `chown`, `USER axiom`), `HEALTHCHECK` against `/api/health` with a 120 s start-period for model loading/migrations, pinned `python:3.11-slim`, minimal `libmagic1` system dep, `.env` and `tests/` excluded via `.dockerignore` (no secrets or test code baked into the image).
- **Compose:** `backend` service under the `fullstack` profile wired to healthy-gated `postgres`/`redis`; credentials are env-required — `API_KEY: ${API_KEY:?…}` preserves W0's fail-closed `:?` enforcement, DB password has no default, model keys are optional-by-design. Duplicate healthcheck present at the compose level. LOG_FORMAT/LOG_LEVEL plumbed through. DEPLOYMENT.md documents the boot sequence and matches the code.

## Criterion 5 — Two-worker smoke proof · **MET (verified statically)**

- `backend/scripts/two_worker_smoke.sh` starts **two independent uvicorn processes** (ports 8901/8902, each `--workers 1`) against one Postgres/Redis, fails fast if Postgres is unreachable, health-waits both workers, POSTs a query through worker A, then asserts worker B serves the trace (`200`, non-empty `trace_steps`, matching `session_id`) via a real JSON assertion — not a grep. Accepts 200 *or* 500 (no-LLM-keys environments persist an error trace through the same write path), rejects everything else as setup failure. Trap-based cleanup; ports/env overridable.
- **Reproducible, not a transcript.** The committed, parameterized script *is* the evidence artifact, with DEPLOYMENT.md instructing re-runs after any trace/eval change. I verified its logic against the serving code (`_persist_trace` write-through on both success and error paths; `_load_trace` PG-first read) but did **not** execute it — this sandbox has no live Postgres/Redis. CI does not run it either (it needs real services); that is the criterion's only soft edge.

## Test quality (claimed 147 passed / 3 skipped locally)

Consistent with what I read: 14 tests in `test_observability.py` + 6 in `test_pg_backed_stores.py` on top of the W1 suite (16 files total), and CI's Backend Tests job is green at the reviewed SHA. Failure paths are genuinely exercised, not just happy paths:

- **Malformed request IDs:** control chars (`\x01`, `\x1f`), 129-char overflow — asserted as rejection + replacement, including the `str.strip` trap specifically.
- **Envelope contract:** exact-shape locks for 401 (W1) and 400 (new), plus a 500 test asserting `code`/`context.request_id` and that a planted DSN/secret never reaches the response body.
- **Migration rollback:** `test_migrations.py` downgrades the full chain (including `c4d8e2f6a9b1`) to base and asserts app tables survive.
- **Cross-worker semantics:** PG-only rows (the second worker's state) served by `/api/trace`, `/api/eval/status`, and `/api/stats`; JSONB string-decode; upsert idempotency (progress tick = 1 row).
- Weaker spots: no dedicated label-cardinality test (bounded-by-construction via route templates is only indirectly tested), and `test_pg_backed_stores.py` mirrors the migration DDL in its fixture — acknowledged drift risk in the fixture docstring.

---

## Top-3 risks (all non-blocking, merged as-is)

1. **Evaluator token blind spot (criterion 2 gap).** Claude-evaluator usage never reaches `record_llm_usage`, so per-query token-cost metrics systematically undercount — worst exactly where cost is highest (evaluation-heavy queries). Cheap fix: route the evaluator's usage through the token scope like `llm/client.py` does.
2. **Metrics correctness is convention, not enforcement (criterion 2).** Single-worker-per-container is documented in two places but nothing stops `--workers N`, which silently yields per-process scrapes. A startup warning when `--workers > 1`, or multiprocess mode, would close it.
3. **Legacy DDL beside the migration chain (criterion 3 caveat).** `vector_store.connect()` still creates `chunk_embeddings` at boot — pre-existing from W1 and deliberately idempotent, but it means "Alembic is the single schema source" has one live exception. Worth consolidating in a W3 schema PR.

## Verdict summary

| Criterion | Verdict |
|---|---|
| 1. JSON logging + request IDs (401 contract intact) | **Met** |
| 2. Prometheus metrics + per-query token accounting | **Met** (evaluator tokens uncounted — named gap) |
| 3. Alembic-only trace-store migration | **Met** (pre-existing `connect()` DDL caveat, unchanged by PR) |
| 4. Dockerfile + compose backend | **Met** |
| 5. Two-worker smoke proof | **Met** (reproducible script; verified statically, not executed here) |

Reviewed SHA: `ee155eab7bbff10ae17d7a8ccf9d17871d86fb57` (base `b4fe841cf34847d642f2f4e0e86c1d1438a3f12b`). PR #4 merged as `0878bfa` before this review completed; findings are advisory for follow-up work.
