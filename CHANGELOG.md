# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-30

### Added

- Initial release.
- Loaders for `.npy`, JSONL, and CSV embedding files with clear errors on
  mismatched dimensions.
- Exact nearest-neighbour search (cosine and euclidean) by id or raw vector.
- Near-duplicate detection: pairwise threshold search plus connected-component
  grouping.
- From-scratch k-means clustering with k-means++ initialisation and a
  silhouette score, including an `--auto-k` mode that scans a k range.
- Outlier detection: points far from every cluster centroid, and points with
  no neighbour above a similarity threshold.
- Drift analysis between two snapshots: centroid shift, per-cluster
  population changes matched across snapshots, and the shared items whose
  local neighbourhood changed the most.
- Dataset statistics: count, dimension, norm distribution, and whether
  vectors are normalised.
- `embed-explorer` CLI with human table and `--json` output for every command.
