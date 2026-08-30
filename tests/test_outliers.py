from __future__ import annotations

import numpy as np
from conftest import make_blobs

from embed_explorer.dataset import EmbeddingSet
from embed_explorer.kmeans import kmeans
from embed_explorer.outliers import centroid_outliers, isolation_outliers


def test_planted_outlier_is_top_centroid_outlier() -> None:
    vectors, _ = make_blobs(centers=[[0, 0]], n_per_blob=20, std=0.1, seed=0)
    outlier = np.array([[100.0, 100.0]])
    all_vectors = np.vstack([vectors, outlier])
    ids = [f"id{i}" for i in range(all_vectors.shape[0])]
    ds = EmbeddingSet(ids=ids, vectors=all_vectors)

    result = kmeans(ds.vectors, k=1, seed=0)
    outliers = centroid_outliers(ds, result.centroids, top_n=1)
    assert outliers[0].id == "id20"


def test_centroid_outliers_ordering_descending() -> None:
    vectors, _ = make_blobs(centers=[[0, 0], [50, 50]], n_per_blob=10, std=0.2, seed=1)
    ids = [f"id{i}" for i in range(vectors.shape[0])]
    ds = EmbeddingSet(ids=ids, vectors=vectors)
    result = kmeans(ds.vectors, k=2, seed=0)
    outliers = centroid_outliers(ds, result.centroids, top_n=5)
    dists = [o.distance_to_nearest_centroid for o in outliers]
    assert dists == sorted(dists, reverse=True)


def test_planted_isolated_point_is_flagged() -> None:
    cluster = np.array([[1.0, 0.0]] * 10)
    isolated = np.array([[0.0, -1.0]])
    vectors = np.vstack([cluster, isolated])
    ids = [f"id{i}" for i in range(vectors.shape[0])]
    ds = EmbeddingSet(ids=ids, vectors=vectors)

    outliers = isolation_outliers(ds, threshold=0.5)
    flagged_ids = {o.id for o in outliers}
    assert "id10" in flagged_ids


def test_no_false_positive_isolation_for_tight_cluster() -> None:
    # Centered away from the origin: cosine similarity near the origin is
    # numerically unstable for tiny perturbations, which is a property of
    # cosine geometry, not a detector bug.
    vectors, _ = make_blobs(centers=[[10, 10]], n_per_blob=10, std=0.01, seed=2)
    ids = [f"id{i}" for i in range(vectors.shape[0])]
    ds = EmbeddingSet(ids=ids, vectors=vectors)
    outliers = isolation_outliers(ds, threshold=0.5)
    assert outliers == []


def test_isolation_outliers_sorted_ascending_by_similarity() -> None:
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(15, 4)) * np.array([1, 1, 1, 1])
    ids = [f"id{i}" for i in range(vectors.shape[0])]
    ds = EmbeddingSet(ids=ids, vectors=vectors)
    outliers = isolation_outliers(ds, threshold=2.0)  # threshold above max sim -> all flagged
    sims = [o.best_neighbor_similarity for o in outliers]
    assert sims == sorted(sims)


def test_isolation_outliers_single_point_returns_empty() -> None:
    ds = EmbeddingSet(ids=["a"], vectors=np.array([[1.0, 2.0]]))
    assert isolation_outliers(ds) == []
