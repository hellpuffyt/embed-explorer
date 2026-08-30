from __future__ import annotations

import numpy as np
import pytest

from embed_explorer.dataset import EmbeddingSet


def make_blobs(
    centers: list[list[float]],
    n_per_blob: int,
    std: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate points around given centers plus the integer label of origin."""
    rng = np.random.default_rng(seed)
    points = []
    labels = []
    for i, center in enumerate(centers):
        c = np.array(center, dtype=np.float64)
        pts = c + rng.normal(scale=std, size=(n_per_blob, len(center)))
        points.append(pts)
        labels.extend([i] * n_per_blob)
    return np.vstack(points), np.array(labels)


@pytest.fixture
def three_blobs() -> tuple[EmbeddingSet, np.ndarray]:
    vectors, labels = make_blobs(
        centers=[[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0]], n_per_blob=20, std=0.3, seed=42
    )
    ids = [f"id{i}" for i in range(vectors.shape[0])]
    return EmbeddingSet(ids=ids, vectors=vectors), labels
