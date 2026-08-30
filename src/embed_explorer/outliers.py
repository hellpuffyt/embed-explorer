"""Outlier detection: far from every centroid, or with no close neighbour."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from embed_explorer.dataset import EmbeddingSet
from embed_explorer.metrics import cosine_similarity_matrix, euclidean_distance_matrix


@dataclass
class CentroidOutlier:
    id: str
    distance_to_nearest_centroid: float


@dataclass
class IsolationOutlier:
    id: str
    best_neighbor_similarity: float


def centroid_outliers(
    dataset: EmbeddingSet,
    centroids: NDArray[np.float64],
    top_n: int = 10,
) -> list[CentroidOutlier]:
    """Points whose distance to their nearest centroid is largest."""
    dist = euclidean_distance_matrix(dataset.vectors, centroids)
    nearest = np.min(dist, axis=1)
    order = np.argsort(-nearest)[:top_n]
    return [
        CentroidOutlier(id=dataset.ids[i], distance_to_nearest_centroid=float(nearest[i]))
        for i in order
    ]


def isolation_outliers(
    dataset: EmbeddingSet,
    threshold: float = 0.5,
) -> list[IsolationOutlier]:
    """Points with no neighbour above ``threshold`` cosine similarity.

    Returned in ascending order of best-neighbour similarity (most isolated first).
    """
    n = dataset.n
    if n < 2:
        return []
    sim = cosine_similarity_matrix(dataset.vectors, dataset.vectors)
    np.fill_diagonal(sim, -np.inf)
    best = np.max(sim, axis=1)
    isolated_idx = np.where(best < threshold)[0]
    order = isolated_idx[np.argsort(best[isolated_idx])]
    return [
        IsolationOutlier(id=dataset.ids[i], best_neighbor_similarity=float(best[i]))
        for i in order
    ]
