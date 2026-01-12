"""Runs the 30-query benchmark against the live AXIOM graph.

Measures: classification accuracy, retrieval strategy accuracy,
RAGAS scores, correction rates, cache behavior, latency.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from axiom.eval_suite.benchmark import BENCHMARK_QUERIES
from axiom.graph.graph import get_graph
from axiom.graph.state import create_initial_state
from axiom.evaluation.critic_llm import critic_llm
from axiom.retrieval.vector_store import vector_store
from axiom.retrieval.bm25_index import bm25_index
from axiom.cache.semantic_cache import semantic_cache

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2]  # backend/

# Max time a single graph invoke may run (classification → retrieval → generate → RAGAS loop).
EVAL_GRAPH_TIMEOUT_SEC = float(os.environ.get("EVAL_GRAPH_TIMEOUT_SEC", "120"))


class EvalRunner:
    """Runs the full 30-query benchmark and computes aggregate metrics."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    async def _ensure_services(self) -> Dict[str, bool]:
        """Connect all services the graph needs. Idempotent — safe to call
        when services are already connected (e.g. via server lifespan)."""
        pg = await vector_store.connect() if not await vector_store.is_connected() else True
        cache = await semantic_cache.connect() if not await semantic_cache.is_connected() else True
        ollama = await critic_llm.connect() if not await critic_llm.is_connected() else True

        if pg:
            from sqlalchemy import text as sa_text
            try:
                async with vector_store._engine.connect() as conn:
                    rows = await conn.execute(sa_text(
                        "SELECT chunk_id, source, content, chunk_index, token_count FROM chunk_embeddings"
                    ))
                    chunks = [dict(r._mapping) for r in rows]
                if chunks and bm25_index.count() == 0:
                    bm25_index.add_chunks(chunks)
                    logger.info("BM25 hydrated — %d chunks", len(chunks))
            except Exception as exc:
                logger.warning("BM25 hydration skipped: %s", exc)

        status = {"pgvector": pg, "redis_cache": cache, "ollama_critic": ollama}
        logger.info("Service status: %s", status)
        return status

    async def run_single(self, benchmark_query: dict, session_id: str) -> dict:
        """Invoke the real graph with one benchmark query and record metrics."""
        query = benchmark_query["query"]
        expected_strategy = benchmark_query["expected_strategy"]
        ground_truth_keywords = benchmark_query.get("ground_truth_keywords", [])

        start = time.perf_counter()
        result: Dict[str, Any] = {
            "query": query,
            "category": benchmark_query.get("category", "unknown"),
            "expected_strategy": expected_strategy,
            "should_trigger_correction": benchmark_query.get("should_trigger_correction", False),
            "actual_strategy": "",
            "ragas_scores": None,
            "correction_attempts": 0,
            "served_from_cache": False,
            "latency_ms": 0.0,
            "is_complete": False,
            "keyword_hit_rate": 0.0,
            "error": None,
        }

        try:
            graph = get_graph()
            initial_state = create_initial_state(user_query=query, session_id=session_id)
            final_state = await asyncio.wait_for(
                graph.ainvoke(
                    initial_state,
                    config={"configurable": {"thread_id": session_id}},
                ),
                timeout=EVAL_GRAPH_TIMEOUT_SEC,
            )

            result["actual_strategy"] = final_state.get("retrieval_strategy", "")
            result["is_complete"] = final_state.get("is_complete", False)
            result["served_from_cache"] = final_state.get("served_from_cache", False)
            result["correction_attempts"] = final_state.get("correction_attempts", 0)

            ragas = final_state.get("ragas_scores")
            if ragas is not None:
                scores = ragas.model_dump() if hasattr(ragas, "model_dump") else dict(ragas)
                result["ragas_scores"] = scores

            final_answer = final_state.get("final_answer", "").lower()
            if ground_truth_keywords and final_answer:
                hits = sum(1 for kw in ground_truth_keywords if kw.lower() in final_answer)
                result["keyword_hit_rate"] = hits / len(ground_truth_keywords)

        except asyncio.TimeoutError:
            result["error"] = f"Query timed out after {EVAL_GRAPH_TIMEOUT_SEC:.0f}s"
            result["is_complete"] = False
            logger.warning("Eval query timed out: %s", query[:60])
        except Exception as exc:
            result["error"] = str(exc)
            logger.error("Eval query failed: %s — %s", query[:60], exc)

        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return result

    async def run_full_suite(self, delay_between_queries: float = 2.0) -> dict:
        """Run all 30 queries sequentially and compute aggregate metrics."""
        await self._ensure_services()
        suite_start = time.perf_counter()
        self.results = []

        for i, bq in enumerate(BENCHMARK_QUERIES, 1):
            session_id = f"eval-{uuid.uuid4().hex[:8]}"
            logger.info("[%d/%d] %s …", i, len(BENCHMARK_QUERIES), bq["query"][:60])
            res = await self.run_single(bq, session_id)
            self.results.append(res)

            if i < len(BENCHMARK_QUERIES):
                await asyncio.sleep(delay_between_queries)

        total_duration_s = time.perf_counter() - suite_start
        return self._compute_aggregate(total_duration_s, self.results)

    def _compute_aggregate(
        self, total_duration_s: float, results: Optional[List[Dict[str, Any]]] = None
    ) -> dict:
        rows = results if results is not None else self.results
        n = len(rows)
        if n == 0:
            return {"total_queries": 0, "error": "no results"}

        completed = [r for r in rows if r["is_complete"]]
        strategy_correct = sum(1 for r in rows if r["actual_strategy"] == r["expected_strategy"])
        corrected = [r for r in rows if r["correction_attempts"] > 0]
        corrected_and_complete = [r for r in corrected if r["is_complete"]]
        cached = [r for r in rows if r["served_from_cache"]]
        latencies = [r["latency_ms"] for r in rows]

        def _safe_avg(values: list) -> float:
            return round(sum(values) / len(values), 4) if values else 0.0

        def _p95(values: list) -> float:
            if not values:
                return 0.0
            s = sorted(values)
            idx = int(len(s) * 0.95)
            return round(s[min(idx, len(s) - 1)], 1)

        faith_scores = []
        rel_scores = []
        ground_scores = []
        composite_scores = []
        for r in rows:
            if r["ragas_scores"]:
                faith_scores.append(r["ragas_scores"].get("faithfulness", 0.0))
                rel_scores.append(r["ragas_scores"].get("answer_relevancy", 0.0))
                ground_scores.append(r["ragas_scores"].get("context_groundedness", 0.0))
                composite_scores.append(r["ragas_scores"].get("composite_score", 0.0))

        scorer_model = "unknown"
        for r in rows:
            if r["ragas_scores"] and r["ragas_scores"].get("scorer_model"):
                scorer_model = r["ragas_scores"]["scorer_model"]
                break

        return {
            "total_queries": n,
            "completed": len(completed),
            "completion_rate": round(len(completed) / n, 4),

            "strategy_accuracy": round(strategy_correct / n, 4),

            "avg_faithfulness": _safe_avg(faith_scores),
            "avg_relevancy": _safe_avg(rel_scores),
            "avg_groundedness": _safe_avg(ground_scores),
            "avg_composite": _safe_avg(composite_scores),

            "correction_rate": round(len(corrected) / n, 4),
            "avg_corrections": _safe_avg([r["correction_attempts"] for r in rows]),
            "correction_success_rate": round(
                len(corrected_and_complete) / len(corrected), 4
            ) if corrected else 0.0,

            "cache_hit_rate": round(len(cached) / n, 4),
            "avg_latency_ms": round(_safe_avg(latencies), 1),
            "p95_latency_ms": _p95(latencies),

            "keyword_hit_rate": _safe_avg([r["keyword_hit_rate"] for r in rows]),

            "scorer_model": scorer_model,
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_duration_minutes": round(total_duration_s / 60, 2),
        }

    def save_results(self, results: dict, path: str = "eval_results.json") -> None:
        """Save full results (aggregate + per-query) to JSON."""
        out_path = RESULTS_DIR / path
        payload = {
            "aggregate": results,
            "per_query": self.results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info("Results saved to %s", out_path)

    async def run_and_print(self) -> dict:
        """Run the suite, print a formatted summary, save results, return aggregate."""
        print("\n" + "=" * 70)
        print("  AXIOM — 30-Query Evaluation Suite")
        print("=" * 70 + "\n")

        aggregate = await self.run_full_suite()

        print("\n" + "-" * 70)
        print("  AGGREGATE RESULTS")
        print("-" * 70)
        rows = [
            ("Total Queries", aggregate["total_queries"]),
            ("Completed", f"{aggregate['completed']}/{aggregate['total_queries']}"),
            ("Completion Rate", f"{aggregate['completion_rate'] * 100:.1f}%"),
            ("", ""),
            ("Strategy Classification Accuracy", f"{aggregate['strategy_accuracy'] * 100:.1f}%"),
            ("", ""),
            ("Avg Faithfulness", f"{aggregate['avg_faithfulness']:.4f}"),
            ("Avg Answer Relevancy", f"{aggregate['avg_relevancy']:.4f}"),
            ("Avg Context Groundedness", f"{aggregate['avg_groundedness']:.4f}"),
            ("Avg Composite RAGAS Score", f"{aggregate['avg_composite']:.4f}"),
            ("", ""),
            ("Correction Rate", f"{aggregate['correction_rate'] * 100:.1f}%"),
            ("Avg Correction Attempts", f"{aggregate['avg_corrections']:.2f}"),
            ("Correction Success Rate", f"{aggregate['correction_success_rate'] * 100:.1f}%"),
            ("", ""),
            ("Cache Hit Rate", f"{aggregate['cache_hit_rate'] * 100:.1f}%"),
            ("Avg Latency", f"{aggregate['avg_latency_ms']:.1f} ms"),
            ("P95 Latency", f"{aggregate['p95_latency_ms']:.1f} ms"),
            ("", ""),
            ("Keyword Hit Rate", f"{aggregate['keyword_hit_rate'] * 100:.1f}%"),
            ("Scorer Model", aggregate["scorer_model"]),
            ("Total Duration", f"{aggregate['total_duration_minutes']:.2f} min"),
        ]

        for label, value in rows:
            if label == "":
                continue
            print(f"  {label:<38} {value}")

        print("-" * 70)

        print("\n  PER-CATEGORY BREAKDOWN:\n")
        categories = {}
        for r in self.results:
            cat = r["category"]
            categories.setdefault(cat, []).append(r)

        for cat, cat_results in categories.items():
            n_cat = len(cat_results)
            n_complete = sum(1 for r in cat_results if r["is_complete"])
            strat_ok = sum(1 for r in cat_results if r["actual_strategy"] == r["expected_strategy"])
            avg_lat = sum(r["latency_ms"] for r in cat_results) / n_cat
            print(f"  {cat:<22} complete={n_complete}/{n_cat}  strategy={strat_ok}/{n_cat}  avg_latency={avg_lat:.0f}ms")

        print("\n" + "=" * 70 + "\n")

        self.save_results(aggregate)
        return aggregate


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(name)s — %(message)s",
    )

    import sys
    import os
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    os.chdir(Path(__file__).resolve().parents[2])

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    runner = EvalRunner()
    asyncio.run(runner.run_and_print())
