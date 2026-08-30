"""Summary statistics for an embedding set."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embed_explorer.dataset import EmbeddingSet


@dataclass
class NormStats:
    min: float
    max: float
    mean: float
    std: float
    median: float


@dataclass
class DatasetStats:
    count: int
    dimension: int
    norms: NormStats
    is_normalized: bool
    n_texts_present: int


def compute_stats(dataset: EmbeddingSet, normalized_tol: float = 1e-3) -> DatasetStats:
    norms = np.linalg.norm(dataset.vectors, axis=1)
    norm_stats = NormStats(
        min=float(norms.min()) if norms.size else 0.0,
        max=float(norms.max()) if norms.size else 0.0,
        mean=float(norms.mean()) if norms.size else 0.0,
        std=float(norms.std()) if norms.size else 0.0,
        median=float(np.median(norms)) if norms.size else 0.0,
    )
    is_normalized = bool(norms.size) and bool(np.allclose(norms, 1.0, atol=normalized_tol))
    n_texts = sum(1 for t in dataset.texts if t is not None)
    return DatasetStats(
        count=dataset.n,
        dimension=dataset.dim,
        norms=norm_stats,
        is_normalized=is_normalized,
        n_texts_present=n_texts,
    )
