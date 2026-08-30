from __future__ import annotations

import numpy as np
import pytest

from embed_explorer.dataset import EmbeddingSet
from embed_explorer.neighbors import nearest_neighbors, nearest_neighbors_by_id


@pytest.fixture
def simple_ds() -> EmbeddingSet:
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    return EmbeddingSet(ids=["a", "b", "c", "d"], vectors=vectors, texts=["A", "B", "C", "D"])


def test_nearest_neighbors_by_id_excludes_self(simple_ds: EmbeddingSet) -> None:
    results = nearest_neighbors_by_id(simple_ds, "a", k=3, metric="cosine")
    assert "a" not in [r.id for r in results]


def test_nearest_neighbors_by_id_closest_is_similar_vector(simple_ds: EmbeddingSet) -> None:
    results = nearest_neighbors_by_id(simple_ds, "a", k=1, metric="cosine")
    assert results[0].id == "b"


def test_nearest_neighbors_cosine_order_descending(simple_ds: EmbeddingSet) -> None:
    results = nearest_neighbors_by_id(simple_ds, "a", k=3, metric="cosine")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_nearest_neighbors_euclidean_order_ascending(simple_ds: EmbeddingSet) -> None:
    results = nearest_neighbors_by_id(simple_ds, "a", k=3, metric="euclidean")
    scores = [r.score for r in results]
    assert scores == sorted(scores)


def test_nearest_neighbors_opposite_vector_is_farthest_cosine(simple_ds: EmbeddingSet) -> None:
    results = nearest_neighbors_by_id(simple_ds, "a", k=3, metric="cosine")
    assert results[-1].id == "d"


def test_nearest_neighbors_by_raw_vector(simple_ds: EmbeddingSet) -> None:
    query = np.array([1.0, 0.0])
    results = nearest_neighbors(simple_ds, query, k=1, metric="cosine")
    assert results[0].id == "a"


def test_nearest_neighbors_wrong_dim_raises(simple_ds: EmbeddingSet) -> None:
    query = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="dimension"):
        nearest_neighbors(simple_ds, query, k=1)


def test_nearest_neighbors_missing_id_raises(simple_ds: EmbeddingSet) -> None:
    from embed_explorer.dataset import LoadError

    with pytest.raises(LoadError, match="not found"):
        nearest_neighbors_by_id(simple_ds, "zzz", k=1)


def test_nearest_neighbors_k_larger_than_dataset(simple_ds: EmbeddingSet) -> None:
    results = nearest_neighbors_by_id(simple_ds, "a", k=100, metric="cosine")
    assert len(results) == 3  # all others, excluding self


def test_nearest_neighbors_carries_text(simple_ds: EmbeddingSet) -> None:
    results = nearest_neighbors_by_id(simple_ds, "a", k=1, metric="cosine")
    assert results[0].text == "B"


def test_nearest_neighbors_invalid_metric_raises(simple_ds: EmbeddingSet) -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        nearest_neighbors_by_id(simple_ds, "a", k=1, metric="manhattan")
