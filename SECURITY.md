# Security Policy

## Supported versions

AXIOM is under active development and ships from `main`. The current released
line is **v1.5**. Security fixes are applied to `main` and rolled forward; we
do not backport to earlier tags.

| Version | Supported |
|---------|-----------|
| 1.5.x   | Yes |
| < 1.5   | No |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

Email the maintainer with details:

- **Contact:** open a private security advisory via
  GitHub → *Security* → *Report a vulnerability* on this repository.
- Include a description, reproduction steps, the affected endpoint or module,
  and the expected impact (information disclosure, RCE, denial of service,
  data corruption, prompt-injection chain, etc.).
- If you have a proof-of-concept, share it as a minimal reproducer.

Expect an acknowledgement within **3 business days** and a remediation plan
or interim mitigation within **14 days** of the initial report. Coordinated
disclosure timelines are negotiated case by case.

## Out of scope

These do not qualify as vulnerabilities:

- Unauthenticated access to a locally-bound `localhost` deployment.
- Rate limits that protect against accidental abuse rather than determined
  attackers — they are denial-of-service mitigations, not authorization.
- Issues that require a malicious operator (you control the server, you
  control the data).
- Behavior of upstream services (Anthropic API, OpenAI API, Tavily, Postgres,
  Redis) — report those upstream.
- Findings that depend on a misconfigured `.env` committed to a public repo.
  That is operator error, not a vulnerability in AXIOM.

## Hardening recommendations for operators

- Set a strong `API_KEY` in production and require it on all `POST`
  endpoints.
- Rotate `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `TAVILY_API_KEY`
  immediately if a secret is exposed.
- Set non-default `POSTGRES_PASSWORD` and `REDIS_PASSWORD` before deploying
  the docker-compose stack to anything reachable from the public internet.
- Put the FastAPI app behind a reverse proxy that handles TLS, request size
  limits, and IP-level rate limiting in addition to the in-app `slowapi`
  limits.
- Restrict `CORS_ORIGINS` to your actual frontend origin in production.
- Treat user-uploaded documents as untrusted input — the ingest path
  validates MIME type and size, but downstream operators should still scan.

## MCP stdio server trust boundary

`backend/mcp_server.py` (Wave 3) exposes AXIOM as a read-only MCP tool over
stdio. Its trust boundary is intentionally narrow:

- **No network listener.** The server opens no sockets — its only transport
  is the stdin/stdout pair its host process created. Anything that can start
  the process can query it; expose it to hosts you trust with corpus-level
  read access. A test pins this (no server/bind/listen surface in the module).
- **Who runs it, owns it.** Unlike the HTTP API, there is no API-key gate
  across the stdio pipe: authentication is "you launched the process". The
  HTTP API's `require_api_key` and rate limits do not apply to MCP sessions.
- **Fail-closed envelopes.** Pipeline failures, invalid configuration, and
  timeouts return sanitized `isError` responses; raw exception detail
  (model names, hosts, credentials) is logged to stderr only and never
  crosses the tool boundary into the client.
- **Read-only by construction.** The single tool, `axiom_query`, runs the
  retrieval pipeline; there is no write, delete, or ingest surface over MCP.

When wiring the server into an MCP host (e.g. Claude Desktop), remember the
host inherits your environment: provider keys in `.env` are readable by the
server process, and answers may quote ingested document content — treat host
selection as corpus access control.

## Disclosure credit

We credit reporters in release notes once a fix is shipped, unless you ask
to remain anonymous.
