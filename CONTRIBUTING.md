# Contributing to AXIOM

Thanks for your interest. This document covers how to set up a development
environment, the expectations for pull requests, and the project's review
process.

## Getting started

1. Fork the repository and clone your fork.
2. Follow the [Installation and Setup](README.md#installation-and-setup)
   instructions in the README to get the backend, frontend, Postgres
   (pgvector), and Redis running locally.
3. Create a feature branch off `main`:

   ```bash
   git checkout -b feat/short-description
   ```

## Development workflow

### Backend

```bash
cd backend
source .venv/bin/activate
pytest tests/ -v
```

All 60 tests must pass before opening a PR. The suite stubs every network call,
so you can run it offline.

### Frontend

```bash
cd frontend
npm install
npm start          # dev server on http://localhost:3000
npm run build      # production build, must succeed before merge
```

The CI pipeline runs `pytest`, `npm run build`, and a check that no source
maps are emitted. Replicate this locally before pushing.

## Pull request checklist

- [ ] Branch is up to date with `main`.
- [ ] Tests pass (`pytest tests/` from `backend/`).
- [ ] Frontend builds clean (`npm run build` from `frontend/`).
- [ ] No secrets, `.env` files, or API keys in the diff.
- [ ] No unrelated changes — keep PRs focused on one feature or fix.
- [ ] Commit messages describe *why* in addition to *what*.
- [ ] If the change affects the user-facing pipeline or API surface, update
      `README.md`.
- [ ] If the change adds a new environment variable, update both
      `.env.example` and the **Environment Variables Reference** table in
      `README.md`.

## Commit message conventions

We use short, imperative commit subjects:

```
feat: add multi-hop decomposition fallback
fix: close AsyncAnthropic httpx client on shutdown
docs: update environment variables reference
test: cover web_search routing after correction loop
refactor: collapse retrieval_*_node duplication
```

Group related changes into one commit when possible. Use the body for the
*why*: constraint, prior incident, design tradeoff.

## Code style

- **Python:** match the existing style. No formatter is enforced, but new code
  should read like the surrounding module. Add type hints where they clarify
  intent.
- **JavaScript:** the React app follows Create React App defaults. Components
  use the existing Tailwind utility classes and the MERIDIAN design tokens
  in `frontend/src/index.css`.
- **Comments:** write *why* comments, not *what* comments. Don't restate the
  code; explain non-obvious constraints or workarounds.

## Reporting issues

Use the **Bug report** issue template for reproducible problems. Include:

- AXIOM version (header / status bar).
- Backend logs around the failure.
- Steps to reproduce, ideally from a clean docker-compose state.
- Whether `USE_CLAUDE_EVALUATOR` is true or false.

Use **Feature request** for new ideas. Explain the use case before the
implementation suggestion — it makes design discussion easier.

## Security disclosures

Do **not** open a public issue for security problems. See
[`SECURITY.md`](SECURITY.md) for the disclosure process.

## Review process

A maintainer reviews each PR. Expect at least one round of feedback. Reviews
focus on:

1. Correctness — does it do what the description claims?
2. Test coverage — are failure modes covered?
3. Pipeline impact — does it preserve the hallucination gate, web fallback,
   and correction loop semantics?
4. README and config drift — are docs aligned with the change?

Merges are squash-only to keep `main` linear.

## License

By contributing, you agree your work is licensed under the
[MIT License](LICENSE) used by the project.
