import pytest

from axiom.cache.semantic_cache import _cosine_similarity


def test_identical_vectors_return_1():
    v = [1.0, 0.0, 0.0]
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_orthogonal_vectors_return_0():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_opposite_vectors_return_negative_1():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert _cosine_similarity(a, b) == pytest.approx(-1.0)


def test_similarity_is_symmetric():
    a = [0.5, 0.3, 0.8]
    b = [0.1, 0.9, 0.4]
    assert _cosine_similarity(a, b) == pytest.approx(_cosine_similarity(b, a))
