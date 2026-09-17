> Source: Obvious artifact `art_Fw8ix4e8` — "[Merged] PR #4: feat(observability): JSON logs, request IDs, Prometheus metrics, Postgres-backed trace store" · exported 2026-09-17. The artifact is a PR-link record; the facts below were verified live from GitHub at export time.

# AXIOM Wave 2 — PR Record

## Pull request

| Field | Value |
|---|---|
| Repository | nihanthnaidu007/AXIOM_Adaptive_RAG |
| PR | [#4](https://github.com/nihanthnaidu007/AXIOM_Adaptive_RAG/pull/4) |
| Title | feat(observability): JSON logs, request IDs, Prometheus metrics, Postgres-backed trace store |
| State | MERGED (2026-09-17T19:26:53Z) |
| Merge commit | `0878bfa5cc5f49f9533ad3645fce8072483c53f6` |

The merge commit is the tip of the Wave 2 chain on `main`:

```
da94af5  feat(api): fail-closed auth, document lifecycle, and cache invalidation (#2)
b4fe841  feat(w1): integration tests, CI gates, alembic single-source schema, sanitized errors (#3)
0878bfa  feat(observability): JSON logs, request IDs, Prometheus metrics, Postgres-backed trace store (#4)
```

## CI evidence (verified on PR #4)

All six checks passed on run `35264548838`:

| Check | Result | Duration |
|---|---|---|
| Backend Ruff Lint | pass | 5s |
| Backend mypy | pass | 2m59s |
| Backend Tests | pass | 2m30s |
| Frontend Build | pass | 49s |
| Frontend ESLint | pass | 33s |
| Alembic fresh-database verification | pass | 1m31s |

Per the delivery brief and the Wave 2 preview review (`w2-preview-review.md` in this folder): backend suite totals 147 passed / 3 skipped on the CI runner.

## Known follow-up gap

The evaluation component's token accounting undercounts actual usage — documented in the Wave 2 preview review as the program's known follow-up gap for AXIOM (not a blocker to the checkpoint).
