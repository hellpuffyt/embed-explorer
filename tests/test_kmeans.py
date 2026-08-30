from __future__ import annotations

import numpy as np
import pytest
from conftest import make_blobs

from embed_explorer.kmeans import best_k, kmeans, silhouette_score


def test_three_well_separated_blobs_yield_three_clusters() -> None:
    vectors, true_labels = make_blobs(
        centers=[[0.0, 0.0], [20.0, 20.0], [-20.0, 20.0]], n_per_blob=15, std=0.2, seed=1
    )
    result = kmeans(vectors, k=3, seed=0)
    # Same number of points per predicted cluster as per true cluster (up to permutation).
    assert sorted(np.bincount(result.labels).tolist()) == sorted(np.bincount(true_labels).tolist())


def test_kmeans_deterministic_given_seed() -> None:
    vectors, _ = make_blobs(centers=[[0, 0], [10, 10]], n_per_blob=10, std=0.3, seed=2)
    r1 = kmeans(vectors, k=2, seed=7)
    r2 = kmeans(vectors, k=2, seed=7)
    np.testing.assert_array_equal(r1.labels, r2.labels)
    np.testing.assert_allclose(r1.centroids, r2.centroids)


def test_kmeans_different_seeds_may_still_converge_to_same_inertia() -> None:
    vectors, _ = make_blobs(centers=[[0, 0], [50, 50]], n_per_blob=10, std=0.1, seed=3)
    r1 = kmeans(vectors, k=2, seed=1)
    r2 = kmeans(vectors, k=2, seed=2)
    assert r1.inertia == pytest.approx(r2.inertia, rel=0.05)


def test_kmeans_k_equals_1() -> None:
    vectors, _ = make_blobs(centers=[[0, 0]], n_per_blob=10, std=0.1, seed=4)
    result = kmeans(vectors, k=1, seed=0)
    assert set(result.labels.tolist()) == {0}
    np.testing.assert_allclose(result.centroids[0], vectors.mean(axis=0), atol=1e-9)


def test_kmeans_k_greater_than_n_raises() -> None:
    vectors = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="cannot exceed"):
        kmeans(vectors, k=5)


def test_kmeans_k_less_than_1_raises() -> None:
    vectors = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError, match="k must be"):
        kmeans(vectors, k=0)


def test_kmeans_handles_duplicate_points_no_nan() -> None:
    vectors = np.array([[1.0, 1.0]] * 5 + [[10.0, 10.0]] * 5)
    result = kmeans(vectors, k=2, seed=0)
    assert np.isfinite(result.centroids).all()
    assert np.bincount(result.labels).tolist() in ([5, 5],)


def test_kmeans_never_produces_empty_cluster() -> None:
    vectors, _ = make_blobs(centers=[[0, 0], [1, 1], [2, 2]], n_per_blob=8, std=0.05, seed=5)
    result = kmeans(vectors, k=5, seed=0)
    counts = np.bincount(result.labels, minlength=5)
    assert np.all(counts > 0)


def test_silhouette_perfect_separation_near_one() -> None:
    vectors, labels = make_blobs(
        centers=[[0, 0], [1000, 1000]], n_per_blob=10, std=0.01, seed=6
    )
    score = silhouette_score(vectors, labels)
    assert score > 0.99


def test_silhouette_single_cluster_is_zero() -> None:
    vectors = np.random.default_rng(0).normal(size=(10, 3))
    labels = np.zeros(10, dtype=np.int64)
    assert silhouette_score(vectors, labels) == 0.0


def test_silhouette_random_labels_lower_than_true_labels() -> None:
    vectors, true_labels = make_blobs(
        centers=[[0, 0], [30, 30], [-30, 30]], n_per_blob=15, std=0.3, seed=7
    )
    true_score = silhouette_score(vectors, true_labels)
    rng = np.random.default_rng(0)
    random_labels = rng.integers(0, 3, size=vectors.shape[0])
    random_score = silhouette_score(vectors, random_labels)
    assert true_score > random_score


def test_best_k_recovers_three_for_three_blobs() -> None:
    vectors, _ = make_blobs(
        centers=[[0, 0], [30, 30], [-30, 30]], n_per_blob=15, std=0.3, seed=8
    )
    results = best_k(vectors, k_min=2, k_max=6, seed=0)
    best = max(results, key=lambda k: results[k][1])
    assert best == 3


def test_kmeans_result_inertia_decreases_with_more_clusters() -> None:
    vectors, _ = make_blobs(centers=[[0, 0], [10, 10], [20, 0]], n_per_blob=10, std=0.3, seed=9)
    r_small = kmeans(vectors, k=1, seed=0)
    r_large = kmeans(vectors, k=3, seed=0)
    assert r_large.inertia < r_small.inertia
