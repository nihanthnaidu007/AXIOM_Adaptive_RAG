#!/usr/bin/env python3
"""Deterministic, LLM-free eval gate for PR CI (Wave 2, D3).

Runs the real retrieval graph against the committed fixture corpus in
``eval/corpus/`` with every model seam stubbed:

- classification  → keyword-rule heuristic mirroring the classify prompt
- generation      → extractive: the answer IS the retrieved context block, so
                    keyword_hit_rate measures retrieval, not a writer model
- RAGAS critic    → fixed passing scores (the correction loop still runs)
- embeddings      → deterministic hash vectors (never leave the process)
- vector store    → forced disconnected (BM25 + fusion + routing stay live)
- reranker        → forced fallback (position-based scores)
- semantic cache  → forced disconnected

The environment therefore behaves identically with or without
PostgreSQL/Redis — the committed thresholds mean the same thing locally and
in CI. Vector-only (ABSTRACT) queries retrieve nothing in this hermetic mode,
which the committed keyword_hit_rate baseline reflects.

The gate asserts ``completion_rate``, ``strategy_accuracy`` and
``keyword_hit_rate`` against ``eval/thresholds.json``; a regression exits 1
and fails the PR. Real RAGAS (live critic + services) runs nightly via the
scheduled workflow — never as a PR gate.

Usage:
  python scripts/eval_gate.py run   --output results.json [--max-queries N]
  python scripts/eval_gate.py check --results results.json [--thresholds t.json]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

CORPUS_DIR = BACKEND_ROOT / "eval" / "corpus"
THRESHOLDS_PATH = BACKEND_ROOT / "eval" / "thresholds.json"
MAX_QUERIES_ENV = "AXIOM_EVAL_GATE_MAX_QUERIES"

_QUERY_TAG_RE = re.compile(r"<user_query>(.*?)</user_query>", re.DOTALL)
_CONTEXT_RE = re.compile(
    r"RETRIEVED (?:DOCUMENT )?CONTEXT \([^)]*\):\n(.*?)(?=\n\nSTRICT RULES:|\n\nPlease answer|\Z)",
    re.DOTALL,
)

# Deterministic stub classifier — mirrors the classify prompt's semantics
# (time markers and compound questions → hybrid, explain/why/conceptual →
# vector, definitions and short entity lookups → bm25, vague chat → hybrid).
_VAGUE_RE = re.compile(
    r"\b(tell me|everything about|how good|what makes this|summarize"
    r"|should i know|about this|what happens|knowledge base)\b"
)
_TIME_RE = re.compile(r"\b(latest|current|recent|modern)\b")
_ABSTRACT_RE = re.compile(r"\b(explain|why|conceptual|meaning)\b")


def heuristic_classify(query: str) -> Dict[str, Any]:
    """Keyword-rule classifier the CI gate runs in place of Claude."""
    q = query.lower()
    if _TIME_RE.search(q) or " vs " in q or _VAGUE_RE.search(q):
        strategy = "hybrid"
    elif q.startswith("how") and " and " in q:
        strategy = "hybrid"
    elif " and " in q and re.search(r"\b(how|why)\b", q.split(" and ", 1)[1]):
        strategy = "hybrid"
    elif _ABSTRACT_RE.search(q) or q.startswith("how"):
        strategy = "vector"
    else:
        strategy = "bm25"
    return {
        "query_type": "multi_hop" if strategy == "hybrid" and " and " in q else "time_sensitive",
        "retrieval_strategy": strategy,
        "reasoning": "Deterministic keyword-rule classifier (eval gate stub)",
        "entities": [],
        "is_multi_hop": " and " in q,
        "sub_queries": [],
    }


def _hash_embedding(text: str, dims: int = 1536) -> List[float]:
    """Stable, content-derived unit vector — no network, fully deterministic."""
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [(seed[i % len(seed)] / 255.0) * 2.0 - 1.0 for i in range(dims)]
    norm = (sum(x * x for x in raw) ** 0.5) or 1.0
    return [x / norm for x in raw]


def _patch(obj: Any, name: str, value: Any) -> None:
    """Install one stub seam (bypasses type checks on patched methods)."""
    setattr(obj, name, value)


def install_stubs() -> None:
    """Replace every model/service seam with deterministic stubs."""
    import axiom.cache.semantic_cache as semantic_cache_module
    import axiom.graph.nodes.classify_query as classify_query
    import axiom.graph.nodes.evaluate_answer as evaluate_answer
    import axiom.graph.nodes.generate_answer as generate_answer
    import axiom.ingest.indexer as indexer_module
    import axiom.retrieval.embeddings as embeddings_module
    import axiom.retrieval.reranker as reranker_module
    import axiom.retrieval.vector_store as vector_store_module

    vector_store = vector_store_module.vector_store
    semantic_cache = semantic_cache_module.semantic_cache

    async def stub_chat_json(prompt: str, model: Any = None, max_tokens: int = 300) -> Dict[str, Any]:
        match = _QUERY_TAG_RE.search(prompt)
        return heuristic_classify(match.group(1) if match else "")

    async def stub_generate(prompt: str, max_tokens: int = 1000, **kwargs: Any) -> str:
        match = _CONTEXT_RE.search(prompt)
        return match.group(1).strip() if match else "INSUFFICIENT_CONTEXT: no retrieved context"

    class StubRagasScorer:
        """Fixed passing scores — the correction loop stays live but never fires."""

        async def score_all(
            self, question: str = "", answer: str = "", chunks: Any = None, **kwargs: Any
        ) -> Dict[str, Any]:
            return {
                "faithfulness": 0.90,
                "answer_relevancy": 0.90,
                "context_groundedness": 0.90,
                "composite_score": 0.90,
                "scorer_model": "ci-stub",
                "parse_error": False,
            }

    async def stub_embed_text(text: str) -> List[float]:
        return _hash_embedding(text)

    async def stub_embed_batch(texts: Any) -> List[List[float]]:
        return [_hash_embedding(t) for t in texts]

    async def stub_search(query_embedding: List[float], top_k: int = 20) -> List[Dict]:
        return []

    async def stub_is_connected() -> bool:
        return False

    async def stub_replace_by_source(source: str, chunks: Any, embeddings: Any) -> int:
        raise RuntimeError("vector store disabled in eval gate")

    _patch(classify_query, "chat_json", stub_chat_json)
    _patch(generate_answer, "generate_with_optional_streaming", stub_generate)
    _patch(evaluate_answer, "ragas_scorer", StubRagasScorer())
    for module in (embeddings_module,):
        _patch(module, "embed_text", stub_embed_text)
    for module_name in (
        "axiom.graph.nodes.check_cache",
        "axiom.graph.nodes.finalize_answer",
        "axiom.graph.nodes.retrieve_hybrid",
        "axiom.graph.nodes.retrieve_vector",
        "axiom.graph.sub_query_runner",
    ):
        import importlib

        _patch(importlib.import_module(module_name), "embed_text", stub_embed_text)
    _patch(indexer_module, "embed_batch", stub_embed_batch)
    _patch(reranker_module.CrossEncoderReranker, "load", lambda self: None)
    _patch(vector_store, "search", stub_search)
    _patch(vector_store, "is_connected", stub_is_connected)
    _patch(vector_store, "replace_by_source", stub_replace_by_source)
    _patch(semantic_cache, "is_connected", stub_is_connected)


async def ingest_corpus(corpus_dir: Path) -> int:
    """Chunk + index the fixture corpus into the BM25 leg (vector leg refuses)."""
    from axiom.ingest.indexer import get_dual_indexer
    from axiom.ingest.loader import DocumentChunker

    chunker = DocumentChunker()
    indexer = get_dual_indexer()
    indexed = 0
    for path in sorted(corpus_dir.glob("*.md")):
        pages = chunker.load_text(path.read_text(encoding="utf-8"), path.name)
        chunks = chunker.chunk(pages, source=path.name)
        result = await indexer.index_chunks(chunks)
        if result.get("bm25") == "indexed":
            indexed += 1
    return indexed


def aggregate_metrics(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-query metrics into the gate's summary (pure)."""
    total = len(per_query)
    if not total:
        return {
            "total_queries": 0,
            "completion_rate": 0.0,
            "strategy_accuracy": 0.0,
            "keyword_hit_rate": 0.0,
        }
    completed = sum(1 for r in per_query if r.get("is_complete"))
    strategy = sum(
        1 for r in per_query if r.get("actual_strategy") == r.get("expected_strategy")
    )
    hits = sum(float(r.get("keyword_hit_rate") or 0.0) for r in per_query)
    composites = [
        float(r["ragas_scores"]["composite_score"])
        for r in per_query
        if isinstance(r.get("ragas_scores"), dict)
        and r["ragas_scores"].get("composite_score") is not None
    ]
    return {
        "total_queries": total,
        "completed": completed,
        "completion_rate": completed / total,
        "strategy_accuracy": strategy / total,
        "keyword_hit_rate": hits / total,
        "avg_composite_score": (sum(composites) / len(composites)) if composites else None,
    }


async def run_suite(output: Path, corpus_dir: Path, max_queries: Optional[int]) -> Dict[str, Any]:
    """Run the stubbed suite over the committed benchmark and write results."""
    install_stubs()

    from axiom.eval_suite.benchmark import BENCHMARK_QUERIES
    from axiom.eval_suite.runner import EvalRunner

    corpus_files = await ingest_corpus(corpus_dir)
    runner = EvalRunner()
    await runner._ensure_services()

    queries = BENCHMARK_QUERIES
    if max_queries is not None:
        queries = queries[:max_queries]

    per_query: List[Dict[str, Any]] = []
    for i, bq in enumerate(queries):
        result = await runner.run_single(bq, session_id=f"eval-gate-{i:03d}")
        per_query.append(result)

    payload = {
        "mode": "stub",
        "corpus_dir": str(corpus_dir),
        "corpus_files": corpus_files,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "aggregate": aggregate_metrics(per_query),
        "per_query": per_query,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def evaluate(aggregate: Dict[str, Any], thresholds: Dict[str, Any]) -> List[str]:
    """Pure threshold check — returns one human-readable string per violation."""
    violations: List[str] = []
    for metric, threshold in thresholds.get("gated_metrics", {}).items():
        value = aggregate.get(metric)
        if value is None:
            violations.append(f"metric '{metric}' missing from results")
        elif value < threshold:
            violations.append(
                f"{metric}={value:.4f} is below the committed threshold {threshold}"
            )
    return violations


def load_thresholds(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run the stubbed suite and write results JSON")
    run_parser.add_argument("--output", required=True, help="path for the results JSON")
    run_parser.add_argument("--corpus", default=str(CORPUS_DIR))
    run_parser.add_argument("--max-queries", type=int, default=None)

    check_parser = sub.add_parser("check", help="assert results against committed thresholds")
    check_parser.add_argument("--results", required=True, help="results JSON from `run`")
    check_parser.add_argument("--thresholds", default=str(THRESHOLDS_PATH))

    args = parser.parse_args(argv)

    if args.command == "run":
        env_cap = os.environ.get(MAX_QUERIES_ENV)
        cap = args.max_queries if args.max_queries is not None else (
            int(env_cap) if env_cap else None
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = asyncio.run(run_suite(output, Path(args.corpus), cap))
        print(json.dumps(payload["aggregate"], indent=2))
        return 0

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    violations = evaluate(results.get("aggregate", {}), load_thresholds(Path(args.thresholds)))
    if violations:
        for v in violations:
            print(f"EVAL GATE FAIL: {v}", file=sys.stderr)
        return 1
    print("EVAL GATE PASS: all gated metrics at or above committed thresholds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
