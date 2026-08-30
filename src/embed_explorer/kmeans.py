"""K-means clustering with k-means++ initialisation and a silhouette score.

Implemented from scratch with numpy so the package has no scikit-learn
dependency; the algorithm and its edge cases (empty clusters, ties, single
cluster) are covered directly by tests instead of trusted to a library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from embed_explorer.metrics import euclidean_distance_matrix


@dataclass
class KMeansResult:
    labels: NDArray[np.int64]
    centroids: NDArray[np.float64]
    inertia: float
    n_iter: int


def _kmeans_plusplus_init(
    vectors: NDArray[np.float64], k: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    n = vectors.shape[0]
    centroids = np.empty((k, vectors.shape[1]), dtype=np.float64)
    first = rng.integers(0, n)
    centroids[0] = vectors[first]
    closest_sq_dist = np.sum((vectors - centroids[0]) ** 2, axis=1)

    for i in range(1, k):
        total = closest_sq_dist.sum()
        if total <= 0:
            # All remaining points coincide with a chosen centroid; pick arbitrarily.
            remaining = rng.integers(0, n)
            centroids[i] = vectors[remaining]
        else:
            probs = closest_sq_dist / total
            chosen = rng.choice(n, p=probs)
            centroids[i] = vectors[chosen]
        new_dist = np.sum((vectors - centroids[i]) ** 2, axis=1)
        closest_sq_dist = np.minimum(closest_sq_dist, new_dist)
    return centroids


def kmeans(
    vectors: NDArray[np.float64],
    k: int,
    seed: int = 0,
    max_iter: int = 300,
    tol: float = 1e-6,
    n_init: int = 5,
) -> KMeansResult:
    """Run k-means with k-means++ initialisation, deterministic given ``seed``.

    Runs ``n_init`` independent initialisations and keeps the lowest-inertia
    result, mirroring common library defaults.
    """
    n = vectors.shape[0]
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed the number of points ({n})")

    rng = np.random.default_rng(seed)
    best: KMeansResult | None = None

    for _init in range(n_init):
        centroids = _kmeans_plusplus_init(vectors, k, rng)
        labels = np.zeros(n, dtype=np.int64)
        n_iter = 0
        for n_iter in range(1, max_iter + 1):  # noqa: B007
            dist = euclidean_distance_matrix(vectors, centroids)
            new_labels = np.argmin(dist, axis=1)

            new_centroids = centroids.copy()
            for c in range(k):
                mask = new_labels == c
                if np.any(mask):
                    new_centroids[c] = vectors[mask].mean(axis=0)
                else:
                    # Re-seed empty cluster at the point farthest from its centroid.
                    farthest = np.argmax(np.min(dist, axis=1))
                    new_centroids[c] = vectors[farthest]

            shift = float(np.linalg.norm(new_centroids - centroids))
            centroids = new_centroids
            labels = new_labels
            if shift < tol:
                break

        final_dist = euclidean_distance_matrix(vectors, centroids)
        inertia = float(np.sum(final_dist[np.arange(n), labels] ** 2))
        if best is None or inertia < best.inertia:
            best = KMeansResult(labels=labels, centroids=centroids, inertia=inertia, n_iter=n_iter)

    assert best is not None
    return best


def silhouette_score(vectors: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    """Mean silhouette coefficient over all points.

    Returns 0.0 when there is only one cluster or every cluster has a single
    point (silhouette is undefined in those cases; 0.0 signals "no verdict").
    """
    n = vectors.shape[0]
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return 0.0

    dist = euclidean_distance_matrix(vectors, vectors)
    scores = np.zeros(n, dtype=np.float64)

    for i in range(n):
        own = labels[i]
        own_mask = (labels == own) & (np.arange(n) != i)
        if not np.any(own_mask):
            scores[i] = 0.0
            continue
        a_i = float(dist[i, own_mask].mean())

        b_i = np.inf
        for other in unique_labels:
            if other == own:
                continue
            other_mask = labels == other
            mean_dist = float(dist[i, other_mask].mean())
            b_i = min(b_i, mean_dist)

        denom = max(a_i, b_i)
        scores[i] = 0.0 if denom == 0 else (b_i - a_i) / denom

    return float(scores.mean())


def best_k(
    vectors: NDArray[np.float64],
    k_min: int,
    k_max: int,
    seed: int = 0,
) -> dict[int, tuple[KMeansResult, float]]:
    """Run k-means for each k in [k_min, k_max] and score with silhouette."""
    n = vectors.shape[0]
    results: dict[int, tuple[KMeansResult, float]] = {}
    for k in range(k_min, min(k_max, n) + 1):
        result = kmeans(vectors, k=k, seed=seed)
        score = silhouette_score(vectors, result.labels) if k > 1 else 0.0
        results[k] = (result, score)
    return results
