"""Distance and similarity primitives shared across the toolkit."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Metric = str  # "cosine" or "euclidean"

VALID_METRICS = ("cosine", "euclidean")


def validate_metric(metric: str) -> None:
    if metric not in VALID_METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {VALID_METRICS}")


def _safe_norms(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1e-12
    return norms


def cosine_similarity_matrix(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Pairwise cosine similarity between rows of a (n, d) and b (m, d)."""
    a_norm = a / _safe_norms(a)[:, None]
    b_norm = b / _safe_norms(b)[:, None]
    return a_norm @ b_norm.T


def euclidean_distance_matrix(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Pairwise Euclidean distance between rows of a (n, d) and b (m, d)."""
    a_sq = np.sum(a**2, axis=1)[:, None]
    b_sq = np.sum(b**2, axis=1)[None, :]
    cross = a @ b.T
    sq = np.maximum(a_sq + b_sq - 2 * cross, 0.0)
    return np.sqrt(sq)


def pairwise_cosine_similarity(
    vectors: NDArray[np.float64],
) -> NDArray[np.float64]:
    return cosine_similarity_matrix(vectors, vectors)
