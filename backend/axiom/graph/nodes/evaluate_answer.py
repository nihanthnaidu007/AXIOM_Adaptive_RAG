"""AXIOM Evaluate Answer Node - Real RAGAS evaluation with Ollama critic fallback."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from axiom.graph.state import RAGASScores, PipelineTraceStep
from axiom.evaluation.critic_llm import critic_llm
from axiom.evaluation.ragas_scorer import ragas_scorer
from axiom.config import get_config

logger = logging.getLogger(__name__)


async def evaluate_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    start_time = datetime.now(timezone.utc)
    cfg = get_config()
    correction_attempts = state.get("correction_attempts", 0)

    ollama_available = await critic_llm.is_connected()

    if not ollama_available:
        logger.warning("Ollama not available — using mock RAGAS scores. Start Ollama to get real evaluation.")
        if correction_attempts == 0:
            mock_faith, mock_rel, mock_ground = 0.52, 0.71, 0.63
        else:
            mock_faith, mock_rel, mock_ground = 0.84, 0.88, 0.79
        composite = round(mock_faith * 0.5 + mock_rel * 0.3 + mock_ground * 0.2, 3)
        below = mock_faith < cfg.faithfulness_threshold
        scores = RAGASScores(
            faithfulness=mock_faith, answer_relevancy=mock_rel,
            context_groundedness=mock_ground, composite_score=composite,
            below_threshold=below, scorer_model="mock",
            evaluation_mode="mock",
        )
        mode = "mock"
    else:
        chunks_content = []
        for c in state.get("reranked_chunks", []):
            if hasattr(c, "content"):
                chunks_content.append(c.content)
            elif isinstance(c, dict):
                chunks_content.append(c.get("content", ""))

        result = await ragas_scorer.score_all(
            question=state.get("user_query", ""),
            answer=state.get("generated_answer", ""),
            chunks=chunks_content,
        )
        has_parse_error = result.get("parse_error", False)
        if has_parse_error:
            # Parse failure: treat as untrusted — cannot pass evaluation
            logger.warning("RAGAS parse error detected — evaluation_mode=parse_error")
            below = True
            mode = "parse_error"
        else:
            below = (
                result["faithfulness"] < cfg.faithfulness_threshold
                or result["answer_relevancy"] < cfg.relevancy_threshold
                or result["context_groundedness"] < cfg.groundedness_threshold
            )
            mode = "real"
        scores = RAGASScores(
            faithfulness=result["faithfulness"],
            answer_relevancy=result["answer_relevancy"],
            context_groundedness=result["context_groundedness"],
            composite_score=result["composite_score"],
            below_threshold=below,
            scorer_model=result["scorer_model"],
            evaluation_mode=mode,
        )

    state["ragas_scores"] = scores
    if "scores_history" not in state or state["scores_history"] is None:
        state["scores_history"] = []
    state["scores_history"].append(scores)

    if mode == "mock" or mode == "parse_error":
        # Mock/parse_error scoring: cannot trust scores, so never claim evaluation passed
        state["hallucination_detected"] = None
        state["evaluation_passed"] = False
    else:
        state["hallucination_detected"] = scores.below_threshold
        state["evaluation_passed"] = not scores.below_threshold

    state["evaluation_mode"] = mode

    end_time = datetime.now(timezone.utc)
    duration_ms = (end_time - start_time).total_seconds() * 1000

    if "trace_steps" not in state or state["trace_steps"] is None:
        state["trace_steps"] = []

    faith_str = f"{scores.faithfulness:.2f}" if scores.faithfulness is not None else "None"
    status_text = "FAILED — hallucination detected" if scores.below_threshold else "PASSED"
    state["trace_steps"].append(PipelineTraceStep(
        node_name="evaluate_answer", status="complete",
        started_at=start_time.isoformat(), duration_ms=round(duration_ms, 2),
        summary=f"RAGAS evaluation {status_text} — faithfulness: {faith_str} ({mode})",
        detail={
            "faithfulness": scores.faithfulness,
            "answer_relevancy": scores.answer_relevancy,
            "context_groundedness": scores.context_groundedness,
            "composite": scores.composite_score,
            "passed": not scores.below_threshold,
            "faithfulness_threshold": cfg.faithfulness_threshold,
            "relevancy_threshold": cfg.relevancy_threshold,
            "groundedness_threshold": cfg.groundedness_threshold,
            "mode": mode,
            "scorer_model": scores.scorer_model,
        },
    ))
    return state
