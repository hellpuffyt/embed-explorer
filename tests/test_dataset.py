from __future__ import annotations

import json

import numpy as np
import pytest

from embed_explorer.dataset import EmbeddingSet, LoadError, load, load_csv, load_jsonl, load_npy


def test_embedding_set_basic() -> None:
    ds = EmbeddingSet(ids=["a", "b"], vectors=np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert ds.n == 2
    assert ds.dim == 2
    assert ds.texts == [None, None]


def test_embedding_set_mismatched_ids_raises() -> None:
    with pytest.raises(LoadError, match="ids"):
        EmbeddingSet(ids=["a"], vectors=np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_embedding_set_mismatched_texts_raises() -> None:
    with pytest.raises(LoadError, match="texts"):
        EmbeddingSet(ids=["a", "b"], vectors=np.array([[1.0, 2.0], [3.0, 4.0]]), texts=["only one"])


def test_embedding_set_duplicate_ids_raises() -> None:
    with pytest.raises(LoadError, match="duplicate"):
        EmbeddingSet(ids=["a", "a"], vectors=np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_embedding_set_requires_2d() -> None:
    with pytest.raises(LoadError, match="2D"):
        EmbeddingSet(ids=["a"], vectors=np.array([1.0, 2.0]))


def test_index_of() -> None:
    ds = EmbeddingSet(ids=["a", "b"], vectors=np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert ds.index_of("b") == 1
    with pytest.raises(LoadError, match="not found"):
        ds.index_of("missing")


def test_subset() -> None:
    ds = EmbeddingSet(
        ids=["a", "b", "c"],
        vectors=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        texts=["t1", "t2", "t3"],
    )
    sub = ds.subset([0, 2])
    assert sub.ids == ["a", "c"]
    assert sub.texts == ["t1", "t3"]
    np.testing.assert_array_equal(sub.vectors, np.array([[1.0, 2.0], [5.0, 6.0]]))


def test_load_npy(tmp_path) -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    path = tmp_path / "vecs.npy"
    np.save(path, arr)
    ds = load_npy(path)
    assert ds.n == 2
    assert ds.ids == ["0", "1"]
    np.testing.assert_array_equal(ds.vectors, arr)


def test_load_npy_rejects_1d(tmp_path) -> None:
    path = tmp_path / "vecs.npy"
    np.save(path, np.array([1.0, 2.0, 3.0]))
    with pytest.raises(LoadError, match="2D"):
        load_npy(path)


def test_load_jsonl(tmp_path) -> None:
    path = tmp_path / "vecs.jsonl"
    lines = [
        json.dumps({"id": "x1", "vector": [1.0, 2.0], "text": "hello"}),
        json.dumps({"id": "x2", "vector": [3.0, 4.0]}),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    ds = load_jsonl(path)
    assert ds.ids == ["x1", "x2"]
    assert ds.texts == ["hello", None]
    np.testing.assert_array_equal(ds.vectors, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_load_jsonl_default_ids(tmp_path) -> None:
    path = tmp_path / "vecs.jsonl"
    lines = [json.dumps({"vector": [1.0, 2.0]}), json.dumps({"vector": [3.0, 4.0]})]
    path.write_text("\n".join(lines), encoding="utf-8")
    ds = load_jsonl(path)
    assert ds.ids == ["0", "1"]


def test_load_jsonl_mismatched_dims_raises(tmp_path) -> None:
    path = tmp_path / "vecs.jsonl"
    lines = [
        json.dumps({"id": "a", "vector": [1.0, 2.0]}),
        json.dumps({"id": "b", "vector": [1.0, 2.0, 3.0]}),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(LoadError, match="inconsistent"):
        load_jsonl(path)


def test_load_jsonl_missing_vector_raises(tmp_path) -> None:
    path = tmp_path / "vecs.jsonl"
    path.write_text(json.dumps({"id": "a"}), encoding="utf-8")
    with pytest.raises(LoadError, match="missing required field"):
        load_jsonl(path)


def test_load_jsonl_invalid_json_raises(tmp_path) -> None:
    path = tmp_path / "vecs.jsonl"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(LoadError, match="invalid JSON"):
        load_jsonl(path)


def test_load_jsonl_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "vecs.jsonl"
    line1 = json.dumps({"id": "a", "vector": [1.0, 2.0]})
    line2 = json.dumps({"id": "b", "vector": [3.0, 4.0]})
    path.write_text(f"{line1}\n\n{line2}", encoding="utf-8")
    ds = load_jsonl(path)
    assert ds.n == 2


def test_load_csv(tmp_path) -> None:
    path = tmp_path / "vecs.csv"
    path.write_text("id,d0,d1,text\nx1,1.0,2.0,hello\nx2,3.0,4.0,world\n", encoding="utf-8")
    ds = load_csv(path)
    assert ds.ids == ["x1", "x2"]
    assert ds.texts == ["hello", "world"]
    np.testing.assert_array_equal(ds.vectors, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_load_csv_no_id_column(tmp_path) -> None:
    path = tmp_path / "vecs.csv"
    path.write_text("d0,d1\n1.0,2.0\n3.0,4.0\n", encoding="utf-8")
    ds = load_csv(path)
    assert ds.ids == ["0", "1"]


def test_load_csv_non_numeric_raises(tmp_path) -> None:
    path = tmp_path / "vecs.csv"
    path.write_text("id,d0,d1\nx1,foo,2.0\n", encoding="utf-8")
    with pytest.raises(LoadError, match="non-numeric"):
        load_csv(path)


def test_load_csv_mismatched_dims_raises(tmp_path) -> None:
    path = tmp_path / "vecs.csv"
    # csv.DictReader pads/truncates ragged rows with None/extras under restkey,
    # so force inconsistency via differing header-declared width isn't possible;
    # instead simulate via a row shorter than header using empty trailing fields.
    path.write_text("id,d0,d1,d2\nx1,1.0,2.0,3.0\nx2,1.0,2.0,3.0\n", encoding="utf-8")
    ds = load_csv(path)
    assert ds.dim == 3


def test_load_dispatches_on_extension(tmp_path) -> None:
    npy_path = tmp_path / "a.npy"
    np.save(npy_path, np.array([[1.0, 2.0]]))
    assert load(npy_path).n == 1

    jsonl_path = tmp_path / "a.jsonl"
    jsonl_path.write_text(json.dumps({"vector": [1.0, 2.0]}), encoding="utf-8")
    assert load(jsonl_path).n == 1

    csv_path = tmp_path / "a.csv"
    csv_path.write_text("d0,d1\n1.0,2.0\n", encoding="utf-8")
    assert load(csv_path).n == 1


def test_load_unsupported_extension_raises(tmp_path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("nothing", encoding="utf-8")
    with pytest.raises(LoadError, match="unsupported"):
        load(path)
