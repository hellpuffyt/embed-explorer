# Production readiness — embed-explorer

Audit date: 2026-10-22. Tested commit: `cb0e67a` (branch `audit/production-readiness`, based on `main` @ `5fd6f8d`).

## Scope

NumPy-only CLI + library for inspecting embedding sets (.npy with `allow_pickle=False`, JSONL, CSV):
exact nearest neighbours, near-duplicate pairs/groups (union-find), k-means++ with silhouette and
auto-k, outliers, snapshot drift, summary stats.

## Platforms

CPython 3.10+, `numpy>=1.24` (tested with NumPy 2.5.4). Verified on macOS 27 (Apple M5), CPython 3.12.14.

## Clean-clone verification (commit `cb0e67a`)

```bash
git clone -b audit/production-readiness <local clone> embed-explorer && cd embed-explorer
python -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"   # done with uv, equivalent
python gen.py   # the snippet from the README
embed-explorer drift old.npy new.npy --k 4 --top 5   # identical to the README output
embed-explorer stats / duplicates / cluster --auto-k   # run OK
pytest && ruff check src tests && mypy                 # 102 passed / clean / clean
```

## Commands and results

| Command | Baseline (`5fd6f8d`) | After (`cb0e67a`) |
| --- | --- | --- |
| `pytest` | exit 0 — 97 passed, 0 failed, 0 skipped | exit 0 — 102 passed, 0 failed, 0 skipped |
| `ruff check .` / `mypy` | exit 0 / exit 0 | exit 0 / exit 0 |
| `uvx pip-audit` | — | no known vulnerabilities |

CI status — default branch: success (`5fd6f8d`, 2026-08-30); audit branch: pending until pushed.

## Math verification (scikit-learn 1.x in a throwaway environment, not a dependency)

50 random datasets (5–200 points, 2–64 dims, scales 0.1–100): silhouette vs
`sklearn.metrics.silhouette_score` max |diff| 6.8e-17; cosine matrix vs `cosine_similarity` max |diff|
8.9e-16; euclidean matrix vs `euclidean_distances` max relative diff 7.3e-8 (expansion-formula
cancellation on near-zero self distances). k-means on `make_blobs(600, 5 centers, 16 dims)`:
adjusted Rand index 1.0. New hand-computed silhouette test (0.8997…).

## Bugs found and fixed

| Severity | Bug | Regression test |
| --- | --- | --- |
| Medium (silent wrong results) | NaN/inf vectors loaded: stats printed `nan`; duplicates/neighbours silently skipped rows | `tests/test_dataset.py::test_non_finite_vectors_are_rejected[*]`, `test_non_finite_jsonl_is_rejected` |
| Docs | README `pip install embed-explorer` (not on PyPI) | README installs from source |
| Docs | README "sample drift output" (12,000 vectors) was not from a real run | replaced by a real run + its generator, reproduced in the clean clone |

## Security review

`.npy` loaded with `allow_pickle=False` (object arrays rejected — verified). No network.
History secret scan: no hits. `pip-audit`: clean. Workflow permissions `contents: read`.

## Known limitations

- Exact O(n²) memory/time for pairwise searches (documented); no ANN index.
- Euclidean distances use the expansion formula (≈1e-8 relative error near zero distance).

## Acceptance checklist

| # | Item | Expected | Evidence | Result |
| --- | --- | --- | --- | --- |
| 1 | Distances correct | = scikit-learn | comparison above | PASS |
| 2 | Silhouette correct | = scikit-learn / hand | comparison + test | PASS |
| 3 | k-means recovers separated clusters | ARI 1.0 | comparison above | PASS |
| 4 | Planted duplicates/outliers found, tight clusters not flagged | existing tests | test suite | PASS |
| 5 | No pickle execution from .npy | rejected | manual check | PASS |
| 6 | Non-finite input rejected | LoadError | new tests | PASS |
| 7 | Clean clone | install/tests/README example | above | PASS |

## Classification proposal

**Production-Ready for Its Documented Scope** (proposal for lead verification).
