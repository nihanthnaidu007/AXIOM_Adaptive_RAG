from axiom.evaluation.thresholds import compute_confidence_band
from axiom.graph.state import RAGASScores


def _make_scores(composite: float, faithfulness: float = 0.9) -> RAGASScores:
    return RAGASScores(
        faithfulness=faithfulness,
        answer_relevancy=0.88,
        context_groundedness=0.85,
        composite_score=composite,
        below_threshold=False,
        scorer_model="test",
        evaluation_mode="real",
    )


def test_verified_band_high_faithfulness():
    scores = _make_scores(composite=0.889, faithfulness=0.90)
    result = compute_confidence_band(scores, correction_attempts=0, served_from_cache=False)
    assert result.label == "VERIFIED"


def test_unreliable_band_low_faithfulness():
    scores = _make_scores(composite=0.30, faithfulness=0.20)
    result = compute_confidence_band(scores, correction_attempts=0, served_from_cache=False)
    assert result.label == "UNRELIABLE"


def test_correction_penalty_applied():
    scores = _make_scores(composite=0.85)
    result = compute_confidence_band(scores, correction_attempts=3, served_from_cache=False)
    assert result.score < 0.85


def test_cache_hit_bonus_applied():
    scores = _make_scores(composite=0.84)
    result = compute_confidence_band(
        scores, correction_attempts=0, served_from_cache=True, cache_similarity=1.0
    )
    assert result.label == "VERIFIED"
