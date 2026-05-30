"""AXIOM Evaluate Answer Node - Real RAGAS evaluation with Claude or Ollama."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from axiom.graph.state import RAGASScores, PipelineTraceStep
from axiom.evaluation.claude_evaluator import claude_evaluator
from axiom.evaluation.ragas_scorer import RAGASScorer, ragas_scorer
from axiom.config import get_config

logger = logging.getLogger(__name__)


async def evaluate_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    start_time = datetime.now(timezone.utc)
    cfg = get_config()
    correction_attempts = state.get("correction_attempts", 0)

    # --- Select evaluator based on config ---
    if cfg.use_claude_evaluator:
        # Default path: use Claude Haiku via Anthropic API.
        # Works in every deployment environment.
        scorer = ragas_scorer
        scores = None
        mode = None
        below = None
    else:
        # Opt-in path: use local Ollama (for developers who have it running).
        from axiom.evaluation.critic_llm import critic_llm
        ollama_available = await critic_llm.is_connected()
        if ollama_available:
            scorer = RAGASScorer(critic=critic_llm, model_name="ollama/llama3.2")
            scores = None
            mode = None
            below = None
        else:
            # Ollama selected but unreachable. Treat as parse_error instead of
            # mock. Unknown quality is failed quality.
            logger.warning(
                "use_claude_evaluator=False but Ollama is unavailable. "
                "Marking evaluation as parse_error. "
                "Set USE_CLAUDE_EVALUATOR=true or start Ollama to get real evaluation."
            )
            scores = RAGASScores(
                faithfulness=None,
                answer_relevancy=None,
                context_groundedness=None,
                composite_score=0.0,
                below_threshold=True,
                scorer_model="ollama/unavailable",
                evaluation_mode="parse_error",
            )
            mode = "parse_error"
            below = True
            scorer = None

    # --- Run evaluation (skip if we already have parse_error scores from above) ---
    if scorer is not None:
        # Build context from document chunks
        chunks_content = []
        for c in state.get("reranked_chunks", []):
            if hasattr(c, "content"):
                chunks_content.append(c.content)
            elif isinstance(c, dict):
                chunks_content.append(c.get("content", ""))

        # When web search was used, include web chunk content so RAGAS
        # evaluates faithfulness against the actual context used for generation.
        # Without this, web-grounded answers are scored against an empty context,
        # producing faithfulness=0.00 regardless of true answer quality.
        if state.get("web_search_used") and not chunks_content:
            for w in state.get("web_search_chunks", []):
                content = w.get("content", "") if isinstance(w, dict) else ""
                if content:
                    chunks_content.append(content)

        result = await scorer.score_all(
            question=state.get("user_query", ""),
            answer=state.get("generated_answer", ""),
            chunks=chunks_content,
        )

        has_parse_error = result.get("parse_error", False)
        if has_parse_error:
            logger.warning("RAGAS parse error — treating as evaluation failed")
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

    if mode == "parse_error":
        # Evaluation failed structurally. Cannot claim quality. Never passes.
        state["hallucination_detected"] = None
        state["evaluation_passed"] = False
    else:
        # "real" mode — trust the scores.
        state["hallucination_detected"] = scores.below_threshold
        state["evaluation_passed"] = not scores.below_threshold

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
            "context_source": "web" if (state.get("web_search_used") and not state.get("reranked_chunks")) else "documents",
            "chunks_evaluated": len(chunks_content) if scorer is not None else 0,
        },
    ))
    return state
