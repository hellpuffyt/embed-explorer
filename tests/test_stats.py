from __future__ import annotations

import numpy as np

from embed_explorer.dataset import EmbeddingSet
from embed_explorer.stats import compute_stats


def test_stats_basic_counts() -> None:
    ds = EmbeddingSet(ids=["a", "b"], vectors=np.array([[3.0, 4.0], [0.0, 5.0]]))
    stats = compute_stats(ds)
    assert stats.count == 2
    assert stats.dimension == 2
    assert stats.norms.min == 5.0
    assert stats.norms.max == 5.0


def test_stats_detects_normalized_vectors() -> None:
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(10, 5))
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ds = EmbeddingSet(ids=[str(i) for i in range(10)], vectors=vecs)
    stats = compute_stats(ds)
    assert stats.is_normalized is True


def test_stats_detects_non_normalized_vectors() -> None:
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(10, 5)) * 5
    ds = EmbeddingSet(ids=[str(i) for i in range(10)], vectors=vecs)
    stats = compute_stats(ds)
    assert stats.is_normalized is False


def test_stats_counts_texts_present() -> None:
    ds = EmbeddingSet(
        ids=["a", "b", "c"],
        vectors=np.array([[1.0], [2.0], [3.0]]),
        texts=["hi", None, "bye"],
    )
    stats = compute_stats(ds)
    assert stats.n_texts_present == 2


def test_stats_norm_stats_mean_matches_manual_computation() -> None:
    vecs = np.array([[3.0, 4.0], [6.0, 8.0]])  # norms 5, 10
    ds = EmbeddingSet(ids=["a", "b"], vectors=vecs)
    stats = compute_stats(ds)
    assert stats.norms.mean == 7.5
    assert stats.norms.median == 7.5
