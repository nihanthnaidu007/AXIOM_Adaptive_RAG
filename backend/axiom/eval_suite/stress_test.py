"""
AXIOM stress test harness — sequential baseline, concurrent load, cache behavior.
Run from backend: python -m axiom.eval_suite.stress_test
"""

from __future__ import annotations

import asyncio
import os
import statistics
from collections import Counter
from datetime import datetime
from typing import Any

import httpx

BACKEND_URL = os.environ.get("AXIOM_STRESS_BACKEND_URL", "http://127.0.0.1:8000")

STRESS_QUERIES = [
    {"query": "What is the BM25 Okapi term frequency formula", "category": "FACTUAL"},
    {"query": "How does scaled dot-product attention work in Transformers", "category": "FACTUAL"},
    {"query": "What is Reciprocal Rank Fusion", "category": "FACTUAL"},
    {"query": "Explain dense passage retrieval", "category": "ABSTRACT"},
    {"query": "How does pgvector store embeddings", "category": "FACTUAL"},
    {"query": "What is RAGAS faithfulness metric", "category": "FACTUAL"},
    {"query": "Compare BM25 vs vector search", "category": "MULTI_HOP"},
    {"query": "How does sentence BERT work", "category": "FACTUAL"},
    {"query": "What is RAG", "category": "ABSTRACT"},
    {"query": "Explain cross-encoder reranking", "category": "ABSTRACT"},
]


async def single_query(
    client: httpx.AsyncClient, query: str, session_id: str
) -> dict[str, Any]:
    import time

    start = time.monotonic()
    try:
        response = await client.post(
            f"{BACKEND_URL}/api/query",
            json={"query": query, "session_id": session_id},
            timeout=180.0,
        )
        latency = (time.monotonic() - start) * 1000
        data = response.json() if response.content else {}
        return {
            "query": query[:40],
            "status": "success" if response.is_success else "error",
            "http_status": response.status_code,
            "latency_ms": round(latency),
            "strategy": data.get("retrieval_strategy"),
            "faithfulness": (data.get("ragas_scores") or {}).get("faithfulness")
            if data.get("ragas_scores")
            else None,
            "is_complete": data.get("is_complete"),
            "correction_attempts": data.get("correction_attempts", 0),
            "sources": [c.get("source") for c in (data.get("reranked_chunks") or [])[:2]],
            "error": None if response.is_success else str(data.get("detail", response.text))[:200],
        }
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return {
            "query": query[:40],
            "status": "error",
            "latency_ms": round(latency),
            "error": str(e)[:200],
        }


def _p95_ms(latencies: list[float]) -> float:
    if not latencies:
        return 0.0
    s = sorted(latencies)
    idx = max(0, min(len(s) - 1, int(round((len(s) - 1) * 0.95))))
    return float(s[idx])


async def run_sequential(queries: list) -> list:
    print("\n=== SEQUENTIAL TEST (baseline) ===")
    results = []
    async with httpx.AsyncClient() as client:
        for i, q in enumerate(queries):
            print(f"  [{i+1}/{len(queries)}] {q['query'][:50]}...", end=" ", flush=True)
            result = await single_query(client, q["query"], f"stress-seq-{i:03d}")
            print(
                f"{result['latency_ms']}ms | {result.get('strategy', 'N/A')} | faith={result.get('faithfulness', 'N/A')}"
            )
            results.append(result)
    return results


async def run_concurrent(queries: list, concurrency: int) -> list:
    print(f"\n=== CONCURRENT TEST (concurrency={concurrency}) ===")
    import time

    start = time.monotonic()

    async with httpx.AsyncClient() as client:
        tasks = [
            single_query(client, q["query"], f"stress-c{concurrency}-{i:03d}")
            for i, q in enumerate(queries[:concurrency])
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    wall_time = (time.monotonic() - start) * 1000
    print(f"  Wall time for {concurrency} concurrent queries: {round(wall_time)}ms")

    clean_results: list = []
    for r in results:
        if isinstance(r, Exception):
            clean_results.append({"status": "exception", "error": str(r)})
        else:
            clean_results.append(r)
            print(
                f"  {r['query'][:40]} | {r['latency_ms']}ms | {r.get('strategy', 'N/A')}"
            )

    return clean_results


async def run_cache_test(query: str, runs: int = 5) -> dict:
    import time

    print(f"\n=== CACHE HIT TEST ({runs} runs of same query) ===")
    latencies: list[float] = []
    cache_hits = 0

    async with httpx.AsyncClient() as client:
        for i in range(runs):
            result = await single_query(client, query, f"stress-cache-{i:03d}")
            latencies.append(float(result["latency_ms"]))
            is_hit = result["latency_ms"] < 500
            if i > 0 and is_hit:
                cache_hits += 1
            print(
                f"  Run {i+1}: {result['latency_ms']}ms {'[CACHE HIT]' if is_hit and i > 0 else '[FRESH]'}"
            )

    rest = latencies[1:]
    return {
        "first_run_ms": int(latencies[0]),
        "cache_hit_avg_ms": round(statistics.mean(rest)) if rest else None,
        "cache_hits": cache_hits,
        "speedup_factor": round(latencies[0] / statistics.mean(rest), 1)
        if rest and statistics.mean(rest) > 0
        else None,
    }


def print_summary(results: list, label: str) -> None:
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]
    latencies = [r["latency_ms"] for r in successful]
    faithfulness_scores = [
        r["faithfulness"] for r in successful if r.get("faithfulness") is not None
    ]

    print(f"\n--- {label} Summary ---")
    print(
        f"  Total: {len(results)} | Success: {len(successful)} | Failed: {len(failed)}"
    )
    if latencies:
        print(
            f"  Latency: avg={round(statistics.mean(latencies))}ms | "
            f"p50={round(statistics.median(latencies))}ms | "
            f"p95={round(_p95_ms([float(x) for x in latencies]))}ms | "
            f"max={max(latencies)}ms"
        )
    if faithfulness_scores:
        print(
            f"  Faithfulness: avg={round(statistics.mean(faithfulness_scores), 3)} | "
            f"min={min(faithfulness_scores)} | max={max(faithfulness_scores)}"
        )
    strategies = [r.get("strategy") for r in successful if r.get("strategy")]
    if strategies:
        print(f"  Strategies: {dict(Counter(strategies))}")
    corrections = [r.get("correction_attempts", 0) for r in successful]
    if corrections:
        print(f"  Avg corrections: {round(statistics.mean(corrections), 2)}")
    sources_seen: set[str] = set()
    for r in successful:
        sources_seen.update(r.get("sources") or [])
    if sources_seen:
        print(f"  Sources seen: {sorted(sources_seen)}")
    if failed:
        errs = [str(r.get("error", "unknown"))[:60] for r in failed]
        print(f"  Errors: {errs}")


async def health_check() -> bool:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{BACKEND_URL}/api/health", timeout=10.0)
            data = r.json()
            print("=== HEALTH CHECK ===")
            print(f"  Status: {data.get('status')}")
            print(f"  BM25: {data.get('index_status', {}).get('bm25_doc_count')} chunks")
            print(f"  pgvector: {data.get('index_status', {}).get('vector_doc_count')} chunks")
            print(f"  postgres: {data.get('services', {}).get('postgres')}")
            print(f"  redis: {data.get('services', {}).get('redis')}")
            print(f"  ollama: {data.get('services', {}).get('ollama')}")

            postgres_ok = data.get("services", {}).get("postgres") == "connected"
            bm25_ok = data.get("index_status", {}).get("bm25_doc_count", 0) > 0

            if not postgres_ok:
                print("  ERROR: postgres not connected — run: docker compose up -d")
                return False
            if not bm25_ok:
                print(
                    "  ERROR: BM25 index empty — start backend after Docker + data load"
                )
                return False

            print("  All systems GO")
            return True
    except Exception as e:
        print(f"  ERROR: Cannot reach server — {e}")
        print("  Run: uvicorn server:app --host 127.0.0.1 --port 8000 (from backend/)")
        return False


async def main() -> None:
    print(f"AXIOM Stress Test — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    ok = await health_check()
    if not ok:
        return

    seq_results = await run_sequential(STRESS_QUERIES[:5])
    print_summary(seq_results, "Sequential (5 queries)")

    conc3_results = await run_concurrent(STRESS_QUERIES, concurrency=3)
    print_summary(conc3_results, "Concurrent-3")

    conc5_results = await run_concurrent(STRESS_QUERIES, concurrency=5)
    print_summary(conc5_results, "Concurrent-5")

    cache_result = await run_cache_test(
        "What is the BM25 Okapi term frequency formula", runs=4
    )
    print("\n--- Cache Test Summary ---")
    print(f"  First run (cold): {cache_result['first_run_ms']}ms")
    print(f"  Cache hit avg: {cache_result['cache_hit_avg_ms']}ms")
    print(f"  Cache hits: {cache_result['cache_hits']}/3")
    print(f"  Speedup factor: {cache_result['speedup_factor']}x")

    edge_queries = [
        {"query": "RAG", "category": "EDGE"},
        {"query": "Tell me everything", "category": "EDGE"},
        {"query": "x" * 500, "category": "EDGE"},
        {"query": "", "category": "EDGE"},
    ]
    print("\n=== EDGE CASE TEST ===")
    async with httpx.AsyncClient() as client:
        for i, q in enumerate(edge_queries):
            text = q["query"][:200] if q["query"] else ""
            if q["query"] == "":
                print("  (empty query) — skipped POST with empty body (422 expected)")
                continue
            result = await single_query(client, text, f"stress-edge-{i:02d}")
            print(
                f"  '{q['query'][:30]}...' → {result['latency_ms']}ms | "
                f"complete={result.get('is_complete')} | error={str(result.get('error', 'none'))[:50]}"
            )

    print("\n" + "=" * 60)
    print("Stress test complete.")


if __name__ == "__main__":
    asyncio.run(main())
