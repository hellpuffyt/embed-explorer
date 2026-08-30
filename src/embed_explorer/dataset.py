"""Loading and representing a set of embeddings."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class LoadError(ValueError):
    """Raised when an embedding file cannot be parsed or is inconsistent."""


@dataclass
class EmbeddingSet:
    """A collection of embedding vectors with optional ids and source text.

    Attributes:
        ids: One id per row, as strings. Defaults to stringified row index.
        vectors: Array of shape (n, d), dtype float64.
        texts: One optional text snippet per row (``None`` where absent).
    """

    ids: list[str]
    vectors: NDArray[np.float64]
    texts: list[str | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise LoadError(f"vectors must be a 2D array, got shape {self.vectors.shape}")
        n = self.vectors.shape[0]
        if len(self.ids) != n:
            raise LoadError(f"got {len(self.ids)} ids but {n} vectors")
        if not self.texts:
            self.texts = [None] * n
        if len(self.texts) != n:
            raise LoadError(f"got {len(self.texts)} texts but {n} vectors")
        if len(set(self.ids)) != len(self.ids):
            dupes = {i for i in self.ids if self.ids.count(i) > 1}
            raise LoadError(f"duplicate ids found: {sorted(dupes)[:5]}")

    @property
    def n(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    def index_of(self, item_id: str) -> int:
        try:
            return self.ids.index(item_id)
        except ValueError as exc:
            raise LoadError(f"id {item_id!r} not found in dataset") from exc

    def subset(self, indices: list[int]) -> EmbeddingSet:
        return EmbeddingSet(
            ids=[self.ids[i] for i in indices],
            vectors=self.vectors[indices],
            texts=[self.texts[i] for i in indices],
        )


def _check_dims_consistent(dims: list[int], source: str) -> int:
    unique = sorted(set(dims))
    if len(unique) > 1:
        raise LoadError(
            f"inconsistent vector dimensions in {source}: found dimensions {unique}. "
            "All vectors in one file must share the same dimensionality."
        )
    return unique[0]


def load_npy(path: str | Path) -> EmbeddingSet:
    """Load embeddings from a .npy file containing a 2D float array.

    Ids default to "0", "1", ... in row order.
    """
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2:
        raise LoadError(f"{path}: expected a 2D array, got shape {arr.shape}")
    arr = arr.astype(np.float64, copy=False)
    ids = [str(i) for i in range(arr.shape[0])]
    return EmbeddingSet(ids=ids, vectors=arr, texts=[None] * arr.shape[0])


def load_jsonl(path: str | Path) -> EmbeddingSet:
    """Load embeddings from JSONL, one object per line.

    Each line must be ``{"id": ..., "vector": [...], "text": ...}``. ``id`` and
    ``text`` are optional; missing ids default to the line number.
    """
    ids: list[str] = []
    vectors: list[list[float]] = []
    texts: list[str | None] = []
    dims: list[int] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise LoadError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if "vector" not in obj:
                raise LoadError(f"{path}:{lineno}: missing required field 'vector'")
            vec = obj["vector"]
            if not isinstance(vec, list) or not vec:
                raise LoadError(f"{path}:{lineno}: 'vector' must be a non-empty list")
            try:
                vecf = [float(x) for x in vec]
            except (TypeError, ValueError) as exc:
                raise LoadError(f"{path}:{lineno}: 'vector' contains non-numeric values") from exc
            dims.append(len(vecf))
            vectors.append(vecf)
            ids.append(str(obj.get("id", lineno - 1)))
            texts.append(obj.get("text"))
    if not vectors:
        raise LoadError(f"{path}: no records found")
    _check_dims_consistent(dims, str(path))
    arr = np.array(vectors, dtype=np.float64)
    return EmbeddingSet(ids=ids, vectors=arr, texts=texts)


def load_csv(path: str | Path) -> EmbeddingSet:
    """Load embeddings from CSV.

    Expected columns: an ``id`` column (optional), an optional ``text`` column,
    and the remaining numeric columns treated as vector dimensions in file order.
    """
    ids: list[str] = []
    vectors: list[list[float]] = []
    texts: list[str | None] = []
    dims: list[int] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise LoadError(f"{path}: empty CSV file")
        fieldnames = list(reader.fieldnames)
        has_id = "id" in fieldnames
        has_text = "text" in fieldnames
        vector_cols = [c for c in fieldnames if c not in ("id", "text")]
        if not vector_cols:
            raise LoadError(f"{path}: no numeric vector columns found")
        for rowno, row in enumerate(reader, start=1):
            try:
                vecf = [float(row[c]) for c in vector_cols]
            except (TypeError, ValueError) as exc:
                raise LoadError(f"{path}: row {rowno}: non-numeric vector value") from exc
            dims.append(len(vecf))
            vectors.append(vecf)
            ids.append(str(row["id"]) if has_id else str(rowno - 1))
            texts.append(row.get("text") if has_text else None)
    if not vectors:
        raise LoadError(f"{path}: no data rows found")
    _check_dims_consistent(dims, str(path))
    arr = np.array(vectors, dtype=np.float64)
    return EmbeddingSet(ids=ids, vectors=arr, texts=texts)


def load(path: str | Path) -> EmbeddingSet:
    """Load an embedding file, dispatching on file extension.

    Supported extensions: ``.npy``, ``.jsonl``, ``.csv``.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        return load_npy(p)
    if suffix == ".jsonl":
        return load_jsonl(p)
    if suffix == ".csv":
        return load_csv(p)
    raise LoadError(f"unsupported file extension {suffix!r} for {p}; use .npy, .jsonl, or .csv")
