"""Exact nearest-neighbour search.

This is brute-force, exact search (O(n) per query) — not an approximate
nearest-neighbour (ANN) index. That is a deliberate choice: it is correct,
simple, and fast enough for the corpus sizes this tool targets (up to a few
hundred thousand vectors). For billion-scale corpora, use a dedicated ANN
library (e.g. FAISS, HNSW) instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from embed_explorer.dataset import EmbeddingSet
from embed_explorer.metrics import (
    cosine_similarity_matrix,
    euclidean_distance_matrix,
    validate_metric,
)


@dataclass
class Neighbor:
    id: str
    score: float
    text: str | None


def nearest_neighbors(
    dataset: EmbeddingSet,
    query: NDArray[np.float64],
    k: int = 10,
    metric: str = "cosine",
    exclude_id: str | None = None,
) -> list[Neighbor]:
    """Return the k nearest neighbours of ``query`` in ``dataset``.

    For cosine, results are sorted by descending similarity (higher is closer).
    For euclidean, results are sorted by ascending distance (lower is closer).
    """
    validate_metric(metric)
    if query.shape != (dataset.dim,):
        raise ValueError(
            f"query vector has dimension {query.shape}, expected ({dataset.dim},)"
        )
    query2d = query.reshape(1, -1)
    if metric == "cosine":
        scores = cosine_similarity_matrix(query2d, dataset.vectors)[0]
        order = np.argsort(-scores)
    else:
        scores = euclidean_distance_matrix(query2d, dataset.vectors)[0]
        order = np.argsort(scores)

    results: list[Neighbor] = []
    for idx in order:
        item_id = dataset.ids[idx]
        if exclude_id is not None and item_id == exclude_id:
            continue
        results.append(Neighbor(id=item_id, score=float(scores[idx]), text=dataset.texts[idx]))
        if len(results) >= k:
            break
    return results


def nearest_neighbors_by_id(
    dataset: EmbeddingSet, item_id: str, k: int = 10, metric: str = "cosine"
) -> list[Neighbor]:
    idx = dataset.index_of(item_id)
    query = dataset.vectors[idx]
    return nearest_neighbors(dataset, query, k=k, metric=metric, exclude_id=item_id)
