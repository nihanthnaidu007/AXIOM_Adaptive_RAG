"""AXIOM Evaluation Thresholds and Confidence Bands - Fully Implemented."""

from axiom.graph.state import RAGASScores, ConfidenceBand

# Scoring thresholds
FAITHFULNESS_THRESHOLD = 0.75
RELEVANCY_THRESHOLD = 0.70
GROUNDEDNESS_THRESHOLD = 0.65


def compute_confidence_band(
    ragas_scores: RAGASScores,
    correction_attempts: int,
    served_from_cache: bool,
    cache_similarity: float = 0.0
) -> ConfidenceBand:
    """
    Compute the confidence band for a final answer.
    
    Score = composite_score - (correction_attempts * 0.10)
    If served_from_cache: score = min(1.0, score + cache_similarity * 0.05)
    
    Bands:
    >= 0.85 -> "VERIFIED"    color_token: "--band-verified"
    >= 0.70 -> "PROBABLE"    color_token: "--band-probable"
    >= 0.55 -> "UNCERTAIN"   color_token: "--band-uncertain"
    <  0.55 -> "UNRELIABLE"  color_token: "--band-unreliable"
    """
    # Base score from RAGAS composite
    score = ragas_scores.composite_score

    # Penalty for correction attempts (each retry reduces confidence).
    # Skip penalty on cache hits — the stored answer already passed evaluation.
    if not served_from_cache:
        score -= correction_attempts * 0.10

    # Bonus for cache hits (cached answers were previously verified)
    if served_from_cache:
        score = min(1.0, score + cache_similarity * 0.05)
    
    # Clamp score to valid range
    score = max(0.0, min(1.0, score))
    
    # Determine band
    if score >= 0.85:
        return ConfidenceBand(
            label="VERIFIED",
            score=round(score, 3),
            color_token="--band-verified",
            reasoning=f"High faithfulness ({ragas_scores.faithfulness:.2f}) with strong grounding" if ragas_scores.faithfulness is not None else "High composite score with strong grounding"
        )
    elif score >= 0.70:
        return ConfidenceBand(
            label="PROBABLE",
            score=round(score, 3),
            color_token="--band-probable",
            reasoning=f"Adequate faithfulness ({ragas_scores.faithfulness:.2f}), some uncertainty in grounding" if ragas_scores.faithfulness is not None else "Adequate composite score, some uncertainty in grounding"
        )
    elif score >= 0.55:
        return ConfidenceBand(
            label="UNCERTAIN",
            score=round(score, 3),
            color_token="--band-uncertain",
            reasoning=f"Moderate faithfulness ({ragas_scores.faithfulness:.2f}), verify claims independently" if ragas_scores.faithfulness is not None else "Moderate composite score, verify claims independently"
        )
    else:
        return ConfidenceBand(
            label="UNRELIABLE",
            score=round(score, 3),
            color_token="--band-unreliable",
            reasoning=f"Low faithfulness ({ragas_scores.faithfulness:.2f}), answer may contain unsupported claims" if ragas_scores.faithfulness is not None else "Low composite score, answer may contain unsupported claims"
        )
