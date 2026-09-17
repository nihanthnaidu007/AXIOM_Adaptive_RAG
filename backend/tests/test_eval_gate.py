"""Eval gate tests (Wave 2, D3).

The gate script (scripts/eval_gate.py) has a pure core — aggregate_metrics
and evaluate — covered in-process here. The full `run` subcommand boots the
real graph with stubs, so that test runs it as a SUBPROCESS: installing the
gate's process-wide stubs inside this pytest process would leak into every
other test module. The threshold-fail case below proves the gate fails
(non-zero) when a metric regresses below the committed thresholds.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

BACKEND_ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = BACKEND_ROOT / "scripts" / "eval_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("eval_gate", GATE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_gate = _load_gate()


def _row(
    is_complete: bool = True,
    strategy_match: bool = True,
    keyword_hit_rate: float = 1.0,
    composite: Any = 0.9,
) -> Dict[str, Any]:
    return {
        "is_complete": is_complete,
        "actual_strategy": "bm25" if strategy_match else "hybrid",
        "expected_strategy": "bm25",
        "keyword_hit_rate": keyword_hit_rate,
        "ragas_scores": {"composite_score": composite} if composite is not None else None,
    }


class TestAggregateMetrics:
    def test_aggregates_a_mixed_run(self):
        rows = [
            _row(),
            _row(keyword_hit_rate=0.5),
            _row(is_complete=False, strategy_match=False, keyword_hit_rate=0.0),
        ]

        agg = eval_gate.aggregate_metrics(rows)

        assert agg["total_queries"] == 3
        assert agg["completed"] == 2
        assert agg["completion_rate"] == 2 / 3
        assert agg["strategy_accuracy"] == 2 / 3
        assert agg["keyword_hit_rate"] == 0.5
        assert agg["avg_composite_score"] == 0.9

    def test_empty_run_yields_zero_metrics(self):
        agg = eval_gate.aggregate_metrics([])

        assert agg["total_queries"] == 0
        assert agg["completion_rate"] == 0.0
        assert agg["strategy_accuracy"] == 0.0
        assert agg["keyword_hit_rate"] == 0.0


class TestEvaluate:
    def test_aggregate_at_thresholds_passes(self):
        aggregate = {"completion_rate": 1.0, "strategy_accuracy": 1.0, "keyword_hit_rate": 0.73}
        thresholds = {"gated_metrics": {"keyword_hit_rate": 0.7}}

        assert eval_gate.evaluate(aggregate, thresholds) == []

    def test_regression_below_threshold_is_reported(self):
        aggregate = {"keyword_hit_rate": 0.5}
        thresholds = {"gated_metrics": {"keyword_hit_rate": 0.7}}

        violations = eval_gate.evaluate(aggregate, thresholds)

        assert len(violations) == 1
        assert "keyword_hit_rate" in violations[0]
        assert "below" in violations[0]

    def test_missing_metric_is_a_violation(self):
        violations = eval_gate.evaluate(
            {}, {"gated_metrics": {"strategy_accuracy": 1.0}}
        )

        assert violations == ["metric 'strategy_accuracy' missing from results"]


class TestCheckCommand:
    def test_threshold_fail_exits_non_zero(self, tmp_path: Path):
        results = {"aggregate": {"keyword_hit_rate": 0.5, "completion_rate": 1.0}}
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results))
        thresholds_path = tmp_path / "thresholds.json"
        thresholds_path.write_text(json.dumps({"gated_metrics": {"keyword_hit_rate": 0.7}}))

        assert eval_gate.main(
            ["check", "--results", str(results_path), "--thresholds", str(thresholds_path)]
        ) == 1

    def test_passing_results_exit_zero(self, tmp_path: Path):
        results = {"aggregate": {"keyword_hit_rate": 0.73, "completion_rate": 1.0}}
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results))
        thresholds_path = tmp_path / "thresholds.json"
        thresholds_path.write_text(json.dumps({"gated_metrics": {"keyword_hit_rate": 0.7}}))

        assert eval_gate.main(
            ["check", "--results", str(results_path), "--thresholds", str(thresholds_path)]
        ) == 0


class TestRunCommand:
    def test_run_produces_deterministic_results(self, tmp_path: Path):
        """Smoke: the real graph, stubbed, over 2 benchmark queries, in a subprocess."""
        output = tmp_path / "results.json"
        proc = subprocess.run(
            [sys.executable, str(GATE_SCRIPT), "run", "--output", str(output), "--max-queries", "2"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(output.read_text())
        assert payload["mode"] == "stub"
        assert payload["aggregate"]["total_queries"] == 2
        assert payload["aggregate"]["completion_rate"] == 1.0
        assert len(payload["per_query"]) == 2

        # Determinism: a second run must be bit-identical modulo latency.
        output2 = tmp_path / "results2.json"
        proc2 = subprocess.run(
            [sys.executable, str(GATE_SCRIPT), "run", "--output", str(output2), "--max-queries", "2"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc2.returncode == 0, proc2.stderr
        first = json.loads(output.read_text())
        second = json.loads(output2.read_text())
        for run in (first, second):
            for row in run["per_query"]:
                row.pop("latency_ms", None)
            run.pop("generated_at", None)
        assert first == second


class TestHeuristicClassifier:
    def test_matches_benchmark_category_semantics(self):
        cases: List[Any] = [
            ("What is the BM25 Okapi term frequency formula", "bm25"),
            ("What are the latest improvements in transformer architecture", "hybrid"),
            ("Explain why dense embeddings capture semantic meaning", "vector"),
            ("How do embeddings work and what role do they play in RAGAS", "hybrid"),
            ("Tell me everything about the system", "hybrid"),
            ("RAG", "bm25"),
        ]
        for query, expected in cases:
            assert eval_gate.heuristic_classify(query)["retrieval_strategy"] == expected
