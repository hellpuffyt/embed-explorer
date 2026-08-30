# embed-explorer

Explore an embedding space from the terminal: nearest neighbours, near-duplicates,
clusters, outliers, and drift between two snapshots — no notebook required.

## What

`embed-explorer` is a command-line tool and a small Python library for inspecting
a set of embedding vectors. Point it at a `.npy`, JSONL, or CSV file of vectors
(with optional ids and source text) and it will tell you:

- what's near what (nearest neighbours, exact search),
- what's redundant (near-duplicate pairs and groups),
- how the space is organised (k-means clustering with a silhouette score),
- what doesn't fit anywhere (outliers), and
- what changed between two snapshots (centroid shift, cluster population
  changes, and which items' neighbourhoods moved the most).

## Why

Embeddings are usually shipped blind: a pipeline builds a vector index and
nobody looks at it again. That means nobody notices that 30% of the chunks in
a RAG corpus are near-duplicates, that one whole cluster is boilerplate, or
that this month's re-embedding drifted away from last month's in ways that
quietly changed retrieval behaviour. `embed-explorer` makes an embedding set
inspectable in a couple of terminal commands, so those problems get caught
before they show up as bad retrieval results in production.

## Features

- **Loaders** for `.npy` (2D float array), JSONL
  (`{"id": ..., "vector": [...], "text": ...}` per line), and CSV
  (`id`, numeric dimension columns, optional `text`). Mismatched
  dimensions across rows raise a clear error instead of a cryptic numpy
  broadcast failure.
- **Nearest neighbours** for an existing id or a raw vector, with cosine or
  euclidean distance. Search is **exact** (brute-force over all vectors), not
  approximate — see [Performance notes](#performance-notes).
- **Near-duplicate detection**: every pair above a similarity (or below a
  distance) threshold, plus those pairs grouped into connected components so
  you can see duplicate clusters, not just pairs.
- **K-means clustering**, implemented from scratch with numpy (k-means++
  initialisation, multiple restarts, empty-cluster re-seeding) plus a
  silhouette score, and an `--auto-k` mode that scans a k range and reports
  the best-scoring one.
- **Outliers**: points far from every cluster centroid, and points with no
  neighbour above a similarity threshold ("isolated" points).
- **Drift between two snapshots**: overall centroid shift, per-cluster
  population change (old clusters matched to new clusters by centroid
  similarity), and the shared items whose local neighbourhood changed the
  most — the fastest way to answer "did this re-embed actually change
  anything?"
- **Stats**: count, dimension, norm distribution (min/max/mean/std/median),
  and whether the vectors are (approximately) unit-normalised.
- **Output**: human-readable tables by default, `--json` for scripting.

## Architecture

```
src/embed_explorer/
  dataset.py     EmbeddingSet + loaders for .npy / .jsonl / .csv
  metrics.py     cosine / euclidean pairwise distance primitives
  neighbors.py   exact nearest-neighbour search
  duplicates.py  pairwise threshold search + union-find grouping
  kmeans.py      k-means++ init, Lloyd's algorithm, silhouette score
  outliers.py    centroid-distance and isolation outlier detection
  drift.py       snapshot comparison: centroid shift, cluster matching,
                 neighbourhood-overlap ranking
  stats.py       summary statistics
  formatting.py  plain-text table rendering
  cli.py         argparse subcommands, wires everything together
```

Everything below `cli.py` is a plain function/dataclass library with no I/O
side effects beyond loading files, so it's usable directly from Python:

```python
from embed_explorer.dataset import load
from embed_explorer.neighbors import nearest_neighbors_by_id

ds = load("chunks.jsonl")
for n in nearest_neighbors_by_id(ds, "chunk-42", k=5):
    print(n.id, n.score, n.text)
```

## Installation

Requires Python 3.10+.

```bash
pip install embed-explorer
```

Or from source:

```bash
git clone https://github.com/hellpuffyt/embed-explorer.git
cd embed-explorer
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"   # Windows
# .venv/bin/python -m pip install -e ".[dev]"            # macOS/Linux
```

Runtime dependency: `numpy` only. No GPU, no network calls, no scikit-learn.

## Usage

```
embed-explorer stats <file> [--json]
embed-explorer neighbors <file> (--id ID | --vector "1.0,2.0,...") [--k N] [--metric cosine|euclidean] [--json]
embed-explorer duplicates <file> [--threshold 0.95] [--metric cosine|euclidean] [--json]
embed-explorer cluster <file> [--k N | --auto-k --k-min N --k-max N] [--seed N] [--json]
embed-explorer outliers <file> [--k N] [--top N] [--isolation-threshold 0.5] [--seed N] [--json]
embed-explorer drift <old_file> <new_file> [--k N] [--neighbors N] [--top N] [--seed N] [--json]
```

## Examples

```bash
# Summary statistics
embed-explorer stats chunks.jsonl

# Nearest neighbours of a known item
embed-explorer neighbors chunks.jsonl --id chunk-42 --k 5

# Nearest neighbours of an arbitrary query vector
embed-explorer neighbors chunks.npy --vector "0.1,0.2,0.3" --k 5 --metric euclidean

# Find near-duplicate chunks in a RAG corpus
embed-explorer duplicates chunks.jsonl --threshold 0.97

# Let the tool pick k for you
embed-explorer cluster chunks.npy --auto-k --k-min 2 --k-max 12

# Fixed k, reproducible
embed-explorer cluster chunks.npy --k 8 --seed 0

# Points that don't belong anywhere
embed-explorer outliers chunks.npy --k 8 --top 20

# Did last month's re-embedding actually change anything?
embed-explorer drift embeddings-2026-07.npy embeddings-2026-08.npy --k 10
```

Sample `drift` output (truncated):

```
old: 12000 vectors, new: 12300 vectors, shared ids: 11800
centroid shift: 0.041288
  old centroid norm: 0.998104
  new centroid norm: 0.997622

cluster population changes (matched by centroid similarity):
old_cluster  new_cluster  old_count  new_count  centroid_similarity
-----------  -----------  ---------  ---------  -------------------
0            0            1532       1489       0.9987
1            3            980        1650       0.9540
...

items whose neighbourhood changed most (lowest overlap first):
id           overlap
-----------  -------
chunk-9931   0.1000
chunk-2210   0.1500
...
```

## Formats

- **`.npy`**: a 2D array of shape `(n, d)`. Ids default to `"0", "1", ...`
  in row order; no text is available.
- **`.jsonl`**: one JSON object per line: `{"id": "...", "vector": [...], "text": "..."}`.
  `id` and `text` are optional; a missing `id` defaults to the line number.
  All `vector` fields in a file must have the same length, or loading fails
  with a clear error naming the mismatched dimensions.
- **`.csv`**: a header row, an optional `id` column, an optional `text`
  column, and the remaining columns treated as vector dimensions in file
  order. A missing `id` column defaults to the row number.

## Testing

```bash
pytest
ruff check .
mypy
```

The test suite uses synthetic, seeded vector sets where the correct answer is
known in advance: well-separated Gaussian blobs that must cluster into the
right number of groups, planted near-duplicate pairs that must be found,
planted outliers that must be flagged, and tight clusters that must *not* be
flagged as duplicates or outliers (false-positive guards).

## Performance notes

Every search in this tool (nearest neighbours, duplicate detection, outlier
detection) is **exact brute-force search**, computing the full pairwise
distance matrix with numpy — it is not an approximate nearest-neighbour (ANN)
index. That's a deliberate tradeoff: the results are always correct, the code
is simple enough to test thoroughly, and it comfortably handles tens of
thousands of vectors on a laptop. For corpora in the millions of vectors, or
for latency-sensitive online serving, use a dedicated ANN library (FAISS,
HNSW, ScaNN, etc.) instead — `embed-explorer` is an inspection tool for
offline analysis, not a serving-time index.

K-means is likewise a from-scratch numpy implementation (no scikit-learn
dependency), run with multiple k-means++ restarts; it scales to tens of
thousands of points but is not tuned for million-point datasets.

## Security

- No network access, no telemetry, no GPU requirement.
- Input files are only ever read, never executed. `.npy` files are loaded
  with `allow_pickle=False`, so a malicious `.npy` file cannot execute
  arbitrary code during loading.
- The only runtime dependency is `numpy`.

## License

MIT — see [LICENSE](LICENSE).
