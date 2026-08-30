from __future__ import annotations

import numpy as np
import pytest
from conftest import make_blobs

from embed_explorer.dataset import EmbeddingSet
from embed_explorer.drift import compute_drift


def _to_ds(vectors: np.ndarray, ids: list[str]) -> EmbeddingSet:
    return EmbeddingSet(ids=ids, vectors=vectors)


def test_identical_snapshots_have_zero_drift() -> None:
    vectors, _ = make_blobs(centers=[[0, 0], [10, 10]], n_per_blob=15, std=0.3, seed=0)
    ids = [f"id{i}" for i in range(vectors.shape[0])]
    old = _to_ds(vectors, ids)
    new = _to_ds(vectors.copy(), ids)
    report = compute_drift(old, new, k_clusters=2, k_neighbors=5, seed=0)
    assert report.centroid_shift == 0.0
    assert all(c.overlap == 1.0 for c in report.neighborhood_changes)


def test_shifted_snapshot_has_nonzero_centroid_shift() -> None:
    vectors, _ = make_blobs(centers=[[0, 0], [10, 10]], n_per_blob=15, std=0.3, seed=1)
    ids = [f"id{i}" for i in range(vectors.shape[0])]
    old = _to_ds(vectors, ids)
    new = _to_ds(vectors + np.array([5.0, 5.0]), ids)
    report = compute_drift(old, new, k_clusters=2, seed=0)
    assert report.centroid_shift == pytest.approx(np.sqrt(50), rel=0.05)


def test_shared_ids_reported_correctly() -> None:
    vectors, _ = make_blobs(centers=[[0, 0]], n_per_blob=10, std=0.1, seed=2)
    old_ids = [f"id{i}" for i in range(10)]
    new_ids = [f"id{i}" for i in range(5, 15)]
    old = _to_ds(vectors, old_ids)
    new = _to_ds(vectors, new_ids)
    report = compute_drift(old, new, k_clusters=1, seed=0)
    assert report.shared_ids == 5


def test_mismatched_dimensions_raise() -> None:
    old = EmbeddingSet(ids=["a"], vectors=np.array([[1.0, 2.0]]))
    new = EmbeddingSet(ids=["a"], vectors=np.array([[1.0, 2.0, 3.0]]))
    with pytest.raises(ValueError, match="mismatched dimensions"):
        compute_drift(old, new)


def test_cluster_population_change_detects_growth() -> None:
    old_vectors, _ = make_blobs(centers=[[0, 0], [50, 50]], n_per_blob=10, std=0.2, seed=3)
    # new snapshot: cluster near [0,0] grows, cluster near [50,50] shrinks
    new_vectors, _ = make_blobs(centers=[[0, 0]], n_per_blob=18, std=0.2, seed=4)
    new_vectors_2, _ = make_blobs(centers=[[50, 50]], n_per_blob=2, std=0.2, seed=5)
    new_vectors = np.vstack([new_vectors, new_vectors_2])
    old_ids = [f"o{i}" for i in range(old_vectors.shape[0])]
    new_ids = [f"n{i}" for i in range(new_vectors.shape[0])]
    old = _to_ds(old_vectors, old_ids)
    new = _to_ds(new_vectors, new_ids)
    report = compute_drift(old, new, k_clusters=2, seed=0)
    counts_changed = [(c.old_count, c.new_count) for c in report.cluster_changes]
    assert len(counts_changed) == 2


def test_neighborhood_change_detects_moved_item() -> None:
    # Two tight clusters far apart; move one shared item from cluster A to cluster B.
    cluster_a = np.tile(np.array([0.0, 0.0]), (5, 1)) + np.random.default_rng(0).normal(
        scale=0.01, size=(5, 2)
    )
    cluster_b = np.tile(np.array([100.0, 100.0]), (5, 1)) + np.random.default_rng(1).normal(
        scale=0.01, size=(5, 2)
    )
    old_vectors = np.vstack([cluster_a, cluster_b])
    ids = [f"id{i}" for i in range(10)]
    old = _to_ds(old_vectors, ids)

    new_vectors = old_vectors.copy()
    new_vectors[0] = np.array([100.0, 100.0])  # id0 moves into cluster b
    new = _to_ds(new_vectors, ids)

    report = compute_drift(old, new, k_clusters=2, k_neighbors=4, top_n_neighborhood=10, seed=0)
    changed_ids = [c.id for c in report.neighborhood_changes if c.overlap < 1.0]
    assert "id0" in changed_ids


def test_no_shared_ids_yields_empty_neighborhood_changes() -> None:
    vectors, _ = make_blobs(centers=[[0, 0]], n_per_blob=5, std=0.1, seed=6)
    old = _to_ds(vectors, [f"o{i}" for i in range(5)])
    new = _to_ds(vectors, [f"n{i}" for i in range(5)])
    report = compute_drift(old, new, k_clusters=1, seed=0)
    assert report.neighborhood_changes == []
