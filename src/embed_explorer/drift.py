"""Drift analysis between two embedding snapshots.

Compares an "old" and a "new" snapshot of (nominally) the same corpus:
overall centroid shift, how cluster populations moved, and which shared
items had their local neighbourhood change the most. This is the feature
that lets someone answer "did last month's re-embedding actually change
anything semantically?" without a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embed_explorer.dataset import EmbeddingSet
from embed_explorer.kmeans import KMeansResult, kmeans
from embed_explorer.metrics import cosine_similarity_matrix


@dataclass
class ClusterPopulationChange:
    old_cluster: int
    new_cluster: int
    old_count: int
    new_count: int
    centroid_similarity: float


@dataclass
class NeighborhoodChange:
    id: str
    overlap: float
    old_neighbors: list[str]
    new_neighbors: list[str]


@dataclass
class DriftReport:
    old_count: int
    new_count: int
    shared_ids: int
    centroid_shift: float
    old_centroid_norm: float
    new_centroid_norm: float
    cluster_changes: list[ClusterPopulationChange]
    neighborhood_changes: list[NeighborhoodChange]


def _overall_centroid_shift(
    old: EmbeddingSet, new: EmbeddingSet
) -> tuple[float, float, float]:
    old_centroid = old.vectors.mean(axis=0)
    new_centroid = new.vectors.mean(axis=0)
    shift = float(np.linalg.norm(old_centroid - new_centroid))
    return shift, float(np.linalg.norm(old_centroid)), float(np.linalg.norm(new_centroid))


def _match_clusters(
    old_result: KMeansResult, new_result: KMeansResult
) -> list[ClusterPopulationChange]:
    """Greedily match old clusters to new clusters by centroid cosine similarity."""
    sim = cosine_similarity_matrix(old_result.centroids, new_result.centroids)
    k_old, k_new = sim.shape
    old_counts = np.bincount(old_result.labels, minlength=k_old)
    new_counts = np.bincount(new_result.labels, minlength=k_new)

    changes: list[ClusterPopulationChange] = []
    used_new: set[int] = set()
    order = sorted(range(k_old), key=lambda i: -old_counts[i])
    for old_idx in order:
        row = sim[old_idx].copy()
        for used in used_new:
            row[used] = -np.inf
        new_idx = int(np.argmax(row))
        used_new.add(new_idx)
        changes.append(
            ClusterPopulationChange(
                old_cluster=old_idx,
                new_cluster=new_idx,
                old_count=int(old_counts[old_idx]),
                new_count=int(new_counts[new_idx]),
                centroid_similarity=float(sim[old_idx, new_idx]),
            )
        )
    return sorted(changes, key=lambda c: c.old_cluster)


def _neighborhood_changes(
    old: EmbeddingSet,
    new: EmbeddingSet,
    shared_ids: list[str],
    k: int,
    top_n: int,
) -> list[NeighborhoodChange]:
    if len(shared_ids) < 2:
        return []
    old_idx = {i: old.index_of(i) for i in shared_ids}
    new_idx = {i: new.index_of(i) for i in shared_ids}

    old_sub = old.vectors[[old_idx[i] for i in shared_ids]]
    new_sub = new.vectors[[new_idx[i] for i in shared_ids]]

    old_sim = cosine_similarity_matrix(old_sub, old_sub)
    new_sim = cosine_similarity_matrix(new_sub, new_sub)
    np.fill_diagonal(old_sim, -np.inf)
    np.fill_diagonal(new_sim, -np.inf)

    kk = min(k, len(shared_ids) - 1)
    results: list[NeighborhoodChange] = []
    for pos, item_id in enumerate(shared_ids):
        old_top = {shared_ids[j] for j in np.argsort(-old_sim[pos])[:kk]}
        new_top = {shared_ids[j] for j in np.argsort(-new_sim[pos])[:kk]}
        union = old_top | new_top
        overlap = len(old_top & new_top) / len(union) if union else 1.0
        results.append(
            NeighborhoodChange(
                id=item_id,
                overlap=overlap,
                old_neighbors=sorted(old_top),
                new_neighbors=sorted(new_top),
            )
        )
    results.sort(key=lambda c: c.overlap)
    return results[:top_n]


def compute_drift(
    old: EmbeddingSet,
    new: EmbeddingSet,
    k_clusters: int = 5,
    k_neighbors: int = 10,
    top_n_neighborhood: int = 10,
    seed: int = 0,
) -> DriftReport:
    if old.dim != new.dim:
        raise ValueError(
            f"snapshots have mismatched dimensions: old={old.dim}, new={new.dim}"
        )
    shift, old_norm, new_norm = _overall_centroid_shift(old, new)

    k_old = min(k_clusters, old.n)
    k_new = min(k_clusters, new.n)
    cluster_changes: list[ClusterPopulationChange] = []
    if k_old >= 1 and k_new >= 1:
        old_result = kmeans(old.vectors, k=k_old, seed=seed)
        new_result = kmeans(new.vectors, k=k_new, seed=seed)
        cluster_changes = _match_clusters(old_result, new_result)

    shared_ids = sorted(set(old.ids) & set(new.ids))
    neighborhood_changes = _neighborhood_changes(
        old, new, shared_ids, k=k_neighbors, top_n=top_n_neighborhood
    )

    return DriftReport(
        old_count=old.n,
        new_count=new.n,
        shared_ids=len(shared_ids),
        centroid_shift=shift,
        old_centroid_norm=old_norm,
        new_centroid_norm=new_norm,
        cluster_changes=cluster_changes,
        neighborhood_changes=neighborhood_changes,
    )


__all__ = [
    "ClusterPopulationChange",
    "NeighborhoodChange",
    "DriftReport",
    "compute_drift",
]
