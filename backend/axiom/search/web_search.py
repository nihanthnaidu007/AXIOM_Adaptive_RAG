"""AXIOM Web Search - Tavily async client wrapper.

Returns normalized result dicts compatible with generate_answer_node's
context block builder. Web chunks are not RetrievedChunk instances because
they have no pgvector metadata (chunk_id, bm25_score, etc.). They are
labeled explicitly in generate_answer_node so the model knows provenance.

Module path: backend/axiom/search/web_search.py
"""

import logging
from typing import Optional

from tavily import AsyncTavilyClient

from axiom.config import get_config

logger = logging.getLogger(__name__)


def is_tavily_configured() -> bool:
    """Return True if TAVILY_API_KEY is non-empty. No network call."""
    return bool(get_config().tavily_api_key)


async def tavily_search(
    query: str,
    search_depth: Optional[str] = None,
    max_results: Optional[int] = None,
) -> list[dict]:
    """Run an async Tavily search and return normalized result dicts.

    Args:
        query:        The search query string.
        search_depth: "basic" or "advanced". Overrides config default when set.
                      "advanced" returns full page content (~2x credit cost).
                      "basic" returns short snippets (~300 chars).
        max_results:  Max results to return. Overrides config default when set.

    Returns:
        List of dicts with keys: url, content, score, title.
        Empty list on any error (missing key, network failure, API error).
        Never raises.
    """
    cfg = get_config()

    if not cfg.tavily_api_key:
        logger.warning(
            "Tavily web search called but TAVILY_API_KEY is not configured. "
            "Set TAVILY_API_KEY in your environment to enable web search fallback."
        )
        return []

    depth = search_depth or cfg.tavily_search_depth
    n = max_results or cfg.tavily_max_results

    try:
        client = AsyncTavilyClient(api_key=cfg.tavily_api_key)

        response = await client.search(
            query=query,
            search_depth=depth,
            max_results=n,
            include_answer=False,
            include_raw_content=False,
        )

        results = response.get("results", [])
        normalized = []
        for r in results:
            content = r.get("content", "").strip()
            if not content:
                continue
            normalized.append({
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": content,
                "score": r.get("score", 0.0),
            })

        logger.info(
            "Tavily search returned %d results (depth=%s, query='%s...')",
            len(normalized), depth, query[:60],
        )
        return normalized

    except ImportError:
        logger.error(
            "tavily-python is not installed. "
            "Run: pip install 'tavily-python>=0.5.0'"
        )
        return []

    except Exception as exc:
        logger.warning(
            "Tavily search failed (%s): %s",
            type(exc).__name__, exc,
        )
        return []
