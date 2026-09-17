"""Prometheus metrics for AXIOM: token-cost accounting + request counters.

Scraped at GET /metrics (unauthenticated, like /health). The token story:

- Generation LLM calls (Anthropic) and embedding calls (OpenAI) report their
  usage through :func:`record_llm_usage`. That mutates the *active token
  scope* — a request-scoped accumulator the /api/query endpoints open around
  the graph run — so per-query cost is attributed to the endpoint that caused
  it. LLM calls outside a scope (startup pings, eval warmups) are dropped
  from per-query accounting by design; the scrape only promises per-request
  and per-endpoint totals.

Exposed families:
- axiom_http_requests_total{method, endpoint, code}
- axiom_query_latency_seconds{endpoint}            (histogram)
- axiom_prompt_tokens_total{endpoint}              (counter)
- axiom_completion_tokens_total{endpoint}          (counter)
- axiom_prompt_tokens_per_query{endpoint}          (histogram)
- axiom_completion_tokens_per_query{endpoint}      (histogram)

The ``endpoint`` label uses the matched route template (``/api/query``), not
the raw path, to bound cardinality.
"""

from contextvars import ContextVar, Token
from typing import Dict, Tuple

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Histogram,
    generate_latest,
)

HTTP_REQUESTS = Counter(
    "axiom_http_requests_total",
    "HTTP requests handled, by method, route template, and status code.",
    ["method", "endpoint", "code"],
)
QUERY_LATENCY = Histogram(
    "axiom_query_latency_seconds",
    "Wall-clock duration of query requests.",
    ["endpoint"],
)
PROMPT_TOKENS = Counter(
    "axiom_prompt_tokens_total",
    "Prompt tokens consumed, by endpoint.",
    ["endpoint"],
)
COMPLETION_TOKENS = Counter(
    "axiom_completion_tokens_total",
    "Completion tokens produced, by endpoint.",
    ["endpoint"],
)
PROMPT_TOKENS_PER_QUERY = Histogram(
    "axiom_prompt_tokens_per_query",
    "Distribution of prompt tokens per query request.",
    ["endpoint"],
)
COMPLETION_TOKENS_PER_QUERY = Histogram(
    "axiom_completion_tokens_per_query",
    "Distribution of completion tokens per query request.",
    ["endpoint"],
)

# The active scope is a mutable dict so record_llm_usage needs no reset;
# nesting replaces the var and end_token_scope restores the parent. The
# default scope absorbs usage recorded outside any request (a no-op: its
# totals are never read).
_token_scope: ContextVar[Dict[str, int]] = ContextVar(
    "axiom_token_scope", default={"prompt": 0, "completion": 0}
)


def record_llm_usage(prompt_tokens: int, completion_tokens: int) -> None:
    """Attribute one LLM/embedding call's usage to the active token scope.

    Called from the LLM and embedding clients right after a successful API
    call. Usage recorded outside a scope lands in the default scope and is
    never exported.
    """
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return
    scope = _token_scope.get()
    scope["prompt"] += max(int(prompt_tokens), 0)
    scope["completion"] += max(int(completion_tokens), 0)


def begin_token_scope() -> "Token[Dict[str, int]]":
    """Open a per-request token scope; pair with :func:`end_token_scope`."""
    return _token_scope.set({"prompt": 0, "completion": 0})


def end_token_scope(token: "Token[Dict[str, int]]") -> Tuple[int, int]:
    """Close a scope and return the (prompt, completion) totals it collected."""
    scope = _token_scope.get()
    _token_scope.reset(token)
    return scope["prompt"], scope["completion"]


def observe_query_tokens(endpoint: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Record a finished query's token usage into the endpoint-labeled families."""
    PROMPT_TOKENS.labels(endpoint=endpoint).inc(prompt_tokens)
    COMPLETION_TOKENS.labels(endpoint=endpoint).inc(completion_tokens)
    PROMPT_TOKENS_PER_QUERY.labels(endpoint=endpoint).observe(prompt_tokens)
    COMPLETION_TOKENS_PER_QUERY.labels(endpoint=endpoint).observe(completion_tokens)


def observe_request(method: str, endpoint: str, code: int, duration_s: float) -> None:
    """Record one completed HTTP request (route template + status)."""
    HTTP_REQUESTS.labels(method=method, endpoint=endpoint, code=str(code)).inc()
    if endpoint.startswith("/api/query"):
        QUERY_LATENCY.labels(endpoint=endpoint).observe(duration_s)


def render_metrics() -> Tuple[bytes, str]:
    """Body + content type for the /metrics scrape response."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
