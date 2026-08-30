from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import make_blobs

from embed_explorer.cli import main


@pytest.fixture
def npy_file(tmp_path):
    vectors, _ = make_blobs(centers=[[0, 0], [10, 10], [-10, 10]], n_per_blob=10, std=0.2, seed=0)
    path = tmp_path / "vecs.npy"
    np.save(path, vectors)
    return path


def test_cli_stats_json(npy_file, capsys) -> None:
    rc = main(["stats", str(npy_file), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["count"] == 30
    assert out["dimension"] == 2


def test_cli_stats_table(npy_file, capsys) -> None:
    rc = main(["stats", str(npy_file)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "count:" in out


def test_cli_neighbors_by_id(npy_file, capsys) -> None:
    rc = main(["neighbors", str(npy_file), "--id", "0", "--k", "3", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out) == 3


def test_cli_neighbors_by_vector(npy_file, capsys) -> None:
    rc = main(["neighbors", str(npy_file), "--vector", "0,0", "--k", "2", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out) == 2


def test_cli_neighbors_requires_id_or_vector(npy_file, capsys) -> None:
    rc = main(["neighbors", str(npy_file)])
    assert rc == 2


def test_cli_duplicates(tmp_path, capsys) -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.0000001], [0.0, 1.0]])
    path = tmp_path / "vecs.npy"
    np.save(path, vectors)
    rc = main(["duplicates", str(path), "--threshold", "0.999", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["pairs"]) >= 1


def test_cli_cluster_fixed_k(npy_file, capsys) -> None:
    rc = main(["cluster", str(npy_file), "--k", "3", "--seed", "0", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["k"] == 3
    assert len(out["cluster_sizes"]) == 3


def test_cli_cluster_auto_k(npy_file, capsys) -> None:
    rc = main(["cluster", str(npy_file), "--auto-k", "--k-min", "2", "--k-max", "6", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["k"] == 3


def test_cli_outliers(npy_file, capsys) -> None:
    rc = main(["outliers", str(npy_file), "--k", "3", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "centroid_outliers" in out
    assert "isolation_outliers" in out


def test_cli_drift(tmp_path, capsys) -> None:
    vectors, _ = make_blobs(centers=[[0, 0], [10, 10]], n_per_blob=10, std=0.2, seed=0)
    old_path = tmp_path / "old.npy"
    new_path = tmp_path / "new.npy"
    np.save(old_path, vectors)
    np.save(new_path, vectors + 1.0)
    rc = main(["drift", str(old_path), str(new_path), "--k", "2", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["centroid_shift"] > 0


def test_cli_load_error_returns_nonzero(tmp_path, capsys) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("nope", encoding="utf-8")
    rc = main(["stats", str(path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err


def test_cli_neighbors_missing_id_returns_error(npy_file, capsys) -> None:
    rc = main(["neighbors", str(npy_file), "--id", "nope"])
    assert rc == 1


def test_cli_help_lists_subcommands(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for cmd in ["stats", "neighbors", "duplicates", "cluster", "outliers", "drift"]:
        assert cmd in out
