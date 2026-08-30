from __future__ import annotations

import numpy as np
import pytest

from embed_explorer.metrics import (
    cosine_similarity_matrix,
    euclidean_distance_matrix,
    pairwise_cosine_similarity,
    validate_metric,
)


def test_validate_metric_ok() -> None:
    validate_metric("cosine")
    validate_metric("euclidean")


def test_validate_metric_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        validate_metric("manhattan")


def test_cosine_identical_vectors_similarity_one() -> None:
    a = np.array([[1.0, 2.0, 3.0]])
    sim = cosine_similarity_matrix(a, a)
    assert sim[0, 0] == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_zero() -> None:
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    sim = cosine_similarity_matrix(a, b)
    assert sim[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_cosine_opposite_vectors_negative_one() -> None:
    a = np.array([[1.0, 0.0]])
    b = np.array([[-1.0, 0.0]])
    sim = cosine_similarity_matrix(a, b)
    assert sim[0, 0] == pytest.approx(-1.0)


def test_cosine_handles_zero_vector_without_nan() -> None:
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 1.0]])
    sim = cosine_similarity_matrix(a, b)
    assert np.isfinite(sim).all()


def test_euclidean_identical_vectors_zero() -> None:
    a = np.array([[1.0, 2.0, 3.0]])
    dist = euclidean_distance_matrix(a, a)
    assert dist[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_euclidean_known_distance() -> None:
    a = np.array([[0.0, 0.0]])
    b = np.array([[3.0, 4.0]])
    dist = euclidean_distance_matrix(a, b)
    assert dist[0, 0] == pytest.approx(5.0)


def test_euclidean_matrix_shape() -> None:
    a = np.random.default_rng(0).normal(size=(4, 3))
    b = np.random.default_rng(1).normal(size=(6, 3))
    dist = euclidean_distance_matrix(a, b)
    assert dist.shape == (4, 6)


def test_pairwise_cosine_similarity_symmetric() -> None:
    a = np.random.default_rng(0).normal(size=(5, 3))
    sim = pairwise_cosine_similarity(a)
    np.testing.assert_allclose(sim, sim.T, atol=1e-9)
    assert np.allclose(np.diag(sim), 1.0)
