from axiom.evaluation.ragas_scorer import _parse_score


def test_parse_score_valid_json():
    assert _parse_score('{"score": 0.85, "reasoning": "good"}') == 0.85


def test_parse_score_clamps_above_1():
    assert _parse_score('{"score": 1.5, "reasoning": "over"}') == 1.0


def test_parse_score_clamps_below_0():
    assert _parse_score('{"score": -0.2, "reasoning": "negative"}') == 0.0


def test_parse_score_malformed_falls_back_to_regex():
    assert _parse_score('some text score is 0.72 done') == 0.72


def test_parse_score_empty_string_returns_none():
    assert _parse_score('') is None


def test_parse_score_no_number_returns_none():
    assert _parse_score('{"error": "failed"}') is None
