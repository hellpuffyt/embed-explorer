"""Command-line interface for embed-explorer."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import numpy as np

from embed_explorer.dataset import LoadError, load
from embed_explorer.drift import compute_drift
from embed_explorer.duplicates import find_duplicate_pairs, group_duplicates
from embed_explorer.formatting import render_table
from embed_explorer.kmeans import KMeansResult, kmeans, silhouette_score
from embed_explorer.neighbors import nearest_neighbors, nearest_neighbors_by_id
from embed_explorer.outliers import centroid_outliers, isolation_outliers
from embed_explorer.stats import compute_stats


def _parse_vector(text: str) -> np.ndarray:
    try:
        return np.array([float(x) for x in text.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise SystemExit(
            f"error: could not parse --vector {text!r} as comma-separated floats"
        ) from exc


def cmd_stats(args: argparse.Namespace) -> int:
    dataset = load(args.file)
    stats = compute_stats(dataset)
    if args.json:
        print(json.dumps(asdict(stats), indent=2))
        return 0
    print(f"count:          {stats.count}")
    print(f"dimension:      {stats.dimension}")
    print(f"texts present:  {stats.n_texts_present}")
    print(f"normalized:     {stats.is_normalized}")
    print("norms:")
    print(f"  min:    {stats.norms.min:.6f}")
    print(f"  max:    {stats.norms.max:.6f}")
    print(f"  mean:   {stats.norms.mean:.6f}")
    print(f"  std:    {stats.norms.std:.6f}")
    print(f"  median: {stats.norms.median:.6f}")
    return 0


def cmd_neighbors(args: argparse.Namespace) -> int:
    dataset = load(args.file)
    if args.id is not None:
        results = nearest_neighbors_by_id(dataset, args.id, k=args.k, metric=args.metric)
    elif args.vector is not None:
        query = _parse_vector(args.vector)
        results = nearest_neighbors(dataset, query, k=args.k, metric=args.metric)
    else:
        print("error: provide either --id or --vector", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
        return 0
    rows = [[r.id, f"{r.score:.6f}", (r.text or "")[:60]] for r in results]
    print(render_table(["id", "score", "text"], rows))
    return 0


def cmd_duplicates(args: argparse.Namespace) -> int:
    dataset = load(args.file)
    pairs = find_duplicate_pairs(dataset, threshold=args.threshold, metric=args.metric)
    groups = group_duplicates(pairs)
    if args.json:
        print(
            json.dumps(
                {
                    "pairs": [asdict(p) for p in pairs],
                    "groups": [asdict(g) for g in groups],
                },
                indent=2,
            )
        )
        return 0
    print(f"{len(pairs)} duplicate pair(s) above threshold {args.threshold} ({args.metric})")
    if pairs:
        rows = [[p.id_a, p.id_b, f"{p.score:.6f}"] for p in pairs]
        print(render_table(["id_a", "id_b", "score"], rows))
    print(f"\n{len(groups)} duplicate group(s)")
    for i, g in enumerate(groups):
        print(f"  group {i}: {', '.join(g.ids)}")
    return 0


def cmd_cluster(args: argparse.Namespace) -> int:
    dataset = load(args.file)
    if args.auto_k:
        best_score: float | None = None
        best_result: KMeansResult | None = None
        best_k_val: int | None = None
        for candidate_k in range(max(2, args.k_min), args.k_max + 1):
            if candidate_k > dataset.n:
                break
            candidate_result = kmeans(dataset.vectors, k=candidate_k, seed=args.seed)
            candidate_score = silhouette_score(dataset.vectors, candidate_result.labels)
            if best_score is None or candidate_score > best_score:
                best_score, best_result, best_k_val = candidate_score, candidate_result, candidate_k
        if best_result is None or best_k_val is None or best_score is None:
            print("error: no valid k in range for this dataset", file=sys.stderr)
            return 2
        result, k, score = best_result, best_k_val, best_score
    else:
        k = args.k
        result = kmeans(dataset.vectors, k=k, seed=args.seed)
        score = silhouette_score(dataset.vectors, result.labels)

    counts = np.bincount(result.labels, minlength=k).tolist()
    if args.json:
        print(
            json.dumps(
                {
                    "k": k,
                    "silhouette_score": score,
                    "inertia": result.inertia,
                    "cluster_sizes": counts,
                    "assignments": {
                        dataset.ids[i]: int(result.labels[i]) for i in range(dataset.n)
                    },
                },
                indent=2,
            )
        )
        return 0
    print(f"k = {k}, silhouette score = {score:.4f}, inertia = {result.inertia:.4f}")
    rows = [[c, counts[c]] for c in range(k)]
    print(render_table(["cluster", "size"], rows))
    return 0


def cmd_outliers(args: argparse.Namespace) -> int:
    dataset = load(args.file)
    k = min(args.k, dataset.n)
    result = kmeans(dataset.vectors, k=k, seed=args.seed)
    centroid_out = centroid_outliers(dataset, result.centroids, top_n=args.top)
    isolation_out = isolation_outliers(dataset, threshold=args.isolation_threshold)

    if args.json:
        print(
            json.dumps(
                {
                    "centroid_outliers": [asdict(o) for o in centroid_out],
                    "isolation_outliers": [asdict(o) for o in isolation_out[: args.top]],
                },
                indent=2,
            )
        )
        return 0
    print(f"Top {len(centroid_out)} centroid outliers (farthest from nearest cluster centroid):")
    print(
        render_table(
            ["id", "distance_to_nearest_centroid"],
            [[o.id, f"{o.distance_to_nearest_centroid:.6f}"] for o in centroid_out],
        )
    )
    print(f"\nIsolated points (no neighbour above cosine similarity {args.isolation_threshold}):")
    top_isolated = isolation_out[: args.top]
    print(
        render_table(
            ["id", "best_neighbor_similarity"],
            [[o.id, f"{o.best_neighbor_similarity:.6f}"] for o in top_isolated],
        )
    )
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    old = load(args.old_file)
    new = load(args.new_file)
    report = compute_drift(
        old,
        new,
        k_clusters=args.k,
        k_neighbors=args.neighbors,
        top_n_neighborhood=args.top,
        seed=args.seed,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "old_count": report.old_count,
                    "new_count": report.new_count,
                    "shared_ids": report.shared_ids,
                    "centroid_shift": report.centroid_shift,
                    "old_centroid_norm": report.old_centroid_norm,
                    "new_centroid_norm": report.new_centroid_norm,
                    "cluster_changes": [asdict(c) for c in report.cluster_changes],
                    "neighborhood_changes": [asdict(c) for c in report.neighborhood_changes],
                },
                indent=2,
            )
        )
        return 0
    print(
        f"old: {report.old_count} vectors, new: {report.new_count} vectors, "
        f"shared ids: {report.shared_ids}"
    )
    print(f"centroid shift: {report.centroid_shift:.6f}")
    print(f"  old centroid norm: {report.old_centroid_norm:.6f}")
    print(f"  new centroid norm: {report.new_centroid_norm:.6f}")
    print("\ncluster population changes (matched by centroid similarity):")
    print(
        render_table(
            ["old_cluster", "new_cluster", "old_count", "new_count", "centroid_similarity"],
            [
                [
                    c.old_cluster,
                    c.new_cluster,
                    c.old_count,
                    c.new_count,
                    f"{c.centroid_similarity:.4f}",
                ]
                for c in report.cluster_changes
            ],
        )
    )
    print("\nitems whose neighbourhood changed most (lowest overlap first):")
    print(
        render_table(
            ["id", "overlap"],
            [[c.id, f"{c.overlap:.4f}"] for c in report.neighborhood_changes],
        )
    )
    return 0


def _add_common_output_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="output JSON instead of a table")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embed-explorer", description="Explore an embedding space from the terminal."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_stats = sub.add_parser("stats", help="summary statistics for an embedding file")
    p_stats.add_argument("file")
    _add_common_output_arg(p_stats)
    p_stats.set_defaults(func=cmd_stats)

    p_neighbors = sub.add_parser("neighbors", help="find nearest neighbours of an id or vector")
    p_neighbors.add_argument("file")
    p_neighbors.add_argument("--id", help="id of an existing item to query from")
    p_neighbors.add_argument("--vector", help="comma-separated raw vector to query with")
    p_neighbors.add_argument("--k", type=int, default=10)
    p_neighbors.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    _add_common_output_arg(p_neighbors)
    p_neighbors.set_defaults(func=cmd_neighbors)

    p_dupes = sub.add_parser("duplicates", help="find near-duplicate pairs and groups")
    p_dupes.add_argument("file")
    p_dupes.add_argument("--threshold", type=float, default=0.95)
    p_dupes.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    _add_common_output_arg(p_dupes)
    p_dupes.set_defaults(func=cmd_duplicates)

    p_cluster = sub.add_parser("cluster", help="k-means clustering with a silhouette score")
    p_cluster.add_argument("file")
    p_cluster.add_argument("--k", type=int, default=5)
    p_cluster.add_argument("--auto-k", action="store_true", help="pick k by best silhouette score")
    p_cluster.add_argument("--k-min", type=int, default=2)
    p_cluster.add_argument("--k-max", type=int, default=10)
    p_cluster.add_argument("--seed", type=int, default=0)
    _add_common_output_arg(p_cluster)
    p_cluster.set_defaults(func=cmd_cluster)

    p_outliers = sub.add_parser(
        "outliers", help="points far from centroids or with no close neighbour"
    )
    p_outliers.add_argument("file")
    p_outliers.add_argument(
        "--k", type=int, default=5, help="number of clusters for centroid outliers"
    )
    p_outliers.add_argument("--top", type=int, default=10)
    p_outliers.add_argument("--isolation-threshold", type=float, default=0.5)
    p_outliers.add_argument("--seed", type=int, default=0)
    _add_common_output_arg(p_outliers)
    p_outliers.set_defaults(func=cmd_outliers)

    p_drift = sub.add_parser("drift", help="compare two embedding snapshots")
    p_drift.add_argument("old_file")
    p_drift.add_argument("new_file")
    p_drift.add_argument("--k", type=int, default=5, help="number of clusters to compare")
    p_drift.add_argument("--neighbors", type=int, default=10, help="neighbourhood size for drift")
    p_drift.add_argument("--top", type=int, default=10)
    p_drift.add_argument("--seed", type=int, default=0)
    _add_common_output_arg(p_drift)
    p_drift.set_defaults(func=cmd_drift)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except LoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
