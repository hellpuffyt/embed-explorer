"""Near-duplicate detection: pairs and connected-component groups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embed_explorer.dataset import EmbeddingSet
from embed_explorer.metrics import (
    cosine_similarity_matrix,
    euclidean_distance_matrix,
    validate_metric,
)


@dataclass
class DuplicatePair:
    id_a: str
    id_b: str
    score: float


@dataclass
class DuplicateGroup:
    ids: list[str]


def find_duplicate_pairs(
    dataset: EmbeddingSet, threshold: float = 0.95, metric: str = "cosine"
) -> list[DuplicatePair]:
    """Return all pairs whose similarity exceeds (cosine) or distance is below
    (euclidean) ``threshold``, sorted by strength descending / distance ascending.
    """
    validate_metric(metric)
    n = dataset.n
    if n < 2:
        return []
    if metric == "cosine":
        sim = cosine_similarity_matrix(dataset.vectors, dataset.vectors)
        iu = np.triu_indices(n, k=1)
        scores = sim[iu]
        mask = scores >= threshold
        order = np.argsort(-scores[mask])
    else:
        dist = euclidean_distance_matrix(dataset.vectors, dataset.vectors)
        iu = np.triu_indices(n, k=1)
        scores = dist[iu]
        mask = scores <= threshold
        order = np.argsort(scores[mask])

    rows = iu[0][mask][order]
    cols = iu[1][mask][order]
    pair_scores = scores[mask][order]
    return [
        DuplicatePair(id_a=dataset.ids[r], id_b=dataset.ids[c], score=float(s))
        for r, c, s in zip(rows, cols, pair_scores, strict=True)
    ]


def group_duplicates(pairs: list[DuplicatePair]) -> list[DuplicateGroup]:
    """Cluster duplicate pairs into connected components (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for pair in pairs:
        union(pair.id_a, pair.id_b)

    groups: dict[str, list[str]] = {}
    for item_id in parent:
        root = find(item_id)
        groups.setdefault(root, []).append(item_id)

    return [DuplicateGroup(ids=sorted(members)) for members in groups.values() if len(members) > 1]
