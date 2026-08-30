from __future__ import annotations

import numpy as np

from embed_explorer.dataset import EmbeddingSet
from embed_explorer.duplicates import DuplicatePair, find_duplicate_pairs, group_duplicates


def test_planted_duplicate_pair_is_found() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 5))
    near_dupe = base + rng.normal(scale=1e-4, size=(1, 5))
    unrelated = rng.normal(size=(5, 5)) * 10
    vectors = np.vstack([base, near_dupe, unrelated])
    ids = [f"id{i}" for i in range(vectors.shape[0])]
    ds = EmbeddingSet(ids=ids, vectors=vectors)

    pairs = find_duplicate_pairs(ds, threshold=0.999, metric="cosine")
    found = {(p.id_a, p.id_b) for p in pairs}
    assert ("id0", "id1") in found


def test_no_false_positive_duplicates_for_distinct_vectors() -> None:
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ds = EmbeddingSet(ids=["a", "b", "c"], vectors=vectors)
    pairs = find_duplicate_pairs(ds, threshold=0.95, metric="cosine")
    assert pairs == []


def test_duplicate_pairs_sorted_by_score_descending() -> None:
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.999, 0.001],
            [0.99, 0.01],
        ]
    )
    ds = EmbeddingSet(ids=["a", "b", "c"], vectors=vectors)
    pairs = find_duplicate_pairs(ds, threshold=0.9, metric="cosine")
    scores = [p.score for p in pairs]
    assert scores == sorted(scores, reverse=True)


def test_duplicate_pairs_euclidean_metric() -> None:
    vectors = np.array([[0.0, 0.0], [0.001, 0.0], [10.0, 10.0]])
    ds = EmbeddingSet(ids=["a", "b", "c"], vectors=vectors)
    pairs = find_duplicate_pairs(ds, threshold=0.01, metric="euclidean")
    found = {(p.id_a, p.id_b) for p in pairs}
    assert ("a", "b") in found
    assert not any("c" in (p.id_a, p.id_b) for p in pairs)


def test_find_duplicate_pairs_single_point_no_error() -> None:
    ds = EmbeddingSet(ids=["a"], vectors=np.array([[1.0, 2.0]]))
    assert find_duplicate_pairs(ds) == []


def test_group_duplicates_transitive_chain() -> None:
    pairs = [
        DuplicatePair(id_a="a", id_b="b", score=0.99),
        DuplicatePair(id_a="b", id_b="c", score=0.98),
    ]
    groups = group_duplicates(pairs)
    assert len(groups) == 1
    assert groups[0].ids == ["a", "b", "c"]


def test_group_duplicates_separate_groups() -> None:
    pairs = [
        DuplicatePair(id_a="a", id_b="b", score=0.99),
        DuplicatePair(id_a="c", id_b="d", score=0.98),
    ]
    groups = group_duplicates(pairs)
    assert len(groups) == 2
    id_sets = {tuple(g.ids) for g in groups}
    assert ("a", "b") in id_sets
    assert ("c", "d") in id_sets


def test_group_duplicates_empty_input() -> None:
    assert group_duplicates([]) == []
