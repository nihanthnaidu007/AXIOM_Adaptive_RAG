"""AXIOM RAGAS Scorer - LLM-based evaluation using Ollama llama3.2 as critic."""

import asyncio
import json
import logging
import re
from typing import Optional

from axiom.evaluation.critic_llm import critic_llm

logger = logging.getLogger(__name__)

_FAITHFULNESS_PROMPT = """You are a faithfulness evaluator. Given an answer and context chunks, score how well the answer is grounded in the context.

Context:
{context}

Answer:
{answer}

Score from 0.0 to 1.0 where:
1.0 = every claim in the answer is directly supported by the context
0.5 = some claims supported, some not
0.0 = answer makes claims not present in the context

Respond with ONLY a JSON object: {{"score": 0.XX, "reasoning": "one sentence"}}"""

_RELEVANCY_PROMPT = """You are an answer relevancy evaluator. Given a question and answer, score how well the answer addresses the question.

Question:
{question}

Answer:
{answer}

Score from 0.0 to 1.0 where:
1.0 = answer directly and completely addresses the question
0.5 = answer partially addresses the question
0.0 = answer does not address the question at all

Respond with ONLY a JSON object: {{"score": 0.XX, "reasoning": "one sentence"}}"""

_GROUNDEDNESS_PROMPT = """You are a context groundedness evaluator. Given an answer and context chunks, score whether the specific facts, numbers, and claims in the answer are traceable to specific passages in the context.

Context:
{context}

Answer:
{answer}

Score from 0.0 to 1.0 where:
1.0 = every fact and claim can be traced to a specific passage
0.5 = some facts traceable, some not
0.0 = no facts are traceable to the context

Respond with ONLY a JSON object: {{"score": 0.XX, "reasoning": "one sentence"}}"""


def _parse_score(raw: str) -> float | None:
    """Extract a float score from LLM JSON output. Returns None on failure."""
    if not raw:
        logger.warning("Parse error: empty LLM response")
        return None
    try:
        match = re.search(r'\{[^}]*"score"\s*:\s*([\d.]+)[^}]*\}', raw)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        data = json.loads(raw.strip())
        score_val = data.get("score")
        if score_val is None:
            logger.warning("Parse error: JSON has no 'score' key. Raw: %s", raw[:200])
            return None
        return max(0.0, min(1.0, float(score_val)))
    except Exception:
        nums = re.findall(r'0\.\d+', raw)
        if nums:
            return max(0.0, min(1.0, float(nums[0])))
        logger.warning("Parse error: could not extract score. Raw: %s", raw[:200])
        return None


class RAGASScorer:
    async def score_faithfulness(self, answer: str, chunks: list[str]) -> float | None:
        context = "\n---\n".join(chunks[:5])
        prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        raw = await critic_llm.generate(prompt, max_tokens=200)
        score = _parse_score(raw)
        if score is not None:
            logger.info("Faithfulness score: %.2f", score)
        else:
            logger.warning("Faithfulness score: parse failed")
        return score

    async def score_answer_relevancy(self, question: str, answer: str) -> float | None:
        prompt = _RELEVANCY_PROMPT.format(question=question, answer=answer)
        raw = await critic_llm.generate(prompt, max_tokens=200)
        score = _parse_score(raw)
        if score is not None:
            logger.info("Answer relevancy score: %.2f", score)
        else:
            logger.warning("Answer relevancy score: parse failed")
        return score

    async def score_context_groundedness(self, answer: str, chunks: list[str]) -> float | None:
        context = "\n---\n".join(chunks[:5])
        prompt = _GROUNDEDNESS_PROMPT.format(context=context, answer=answer)
        raw = await critic_llm.generate(prompt, max_tokens=200)
        score = _parse_score(raw)
        if score is not None:
            logger.info("Context groundedness score: %.2f", score)
        else:
            logger.warning("Context groundedness score: parse failed")
        return score

    async def score_all(
        self, question: str, answer: str, chunks: list[str]
    ) -> dict:
        faith, rel, ground = await asyncio.gather(
            self.score_faithfulness(answer, chunks),
            self.score_answer_relevancy(question, answer),
            self.score_context_groundedness(answer, chunks),
        )
        has_parse_error = faith is None or rel is None or ground is None
        # Use 0.0 for composite calculation when a score failed to parse
        f = faith if faith is not None else 0.0
        r = rel if rel is not None else 0.0
        g = ground if ground is not None else 0.0
        composite = round(f * 0.5 + r * 0.3 + g * 0.2, 4)
        return {
            "faithfulness": round(faith, 4) if faith is not None else None,
            "answer_relevancy": round(rel, 4) if rel is not None else None,
            "context_groundedness": round(ground, 4) if ground is not None else None,
            "composite_score": composite,
            "scorer_model": "ollama/llama3.2",
            "parse_error": has_parse_error,
        }


ragas_scorer = RAGASScorer()
