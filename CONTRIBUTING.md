# Contributing

Thanks for considering a contribution to embed-explorer.

## Setup

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"   # Windows
# or: .venv/bin/python -m pip install -e ".[dev]"        # macOS/Linux
```

## Workflow

1. Make your change with tests. Synthetic, seeded test data (well-separated
   blobs, planted duplicates, planted outliers) is strongly preferred over
   real embedding dumps — it lets a test assert the *correct* answer.
2. Run the full gate before opening a pull request:

   ```bash
   pytest
   ruff check .
   mypy
   ```

3. Keep the runtime dependency footprint at `numpy` only. Anything else
   (scikit-learn, pandas, etc.) belongs in `[project.optional-dependencies]`
   at most, and should have a strong justification — the point of this tool
   is that it installs in seconds with no GPU or network requirement.
4. Match the existing style: dataclasses for result types, pure functions in
   the library layer, argument parsing and printing confined to `cli.py`.

## Reporting bugs

Open an issue with the command you ran, the input file shape (rows,
dimensions), and the actual vs. expected output. A minimal reproducible
`.npy`/`.jsonl` fixture is extremely helpful.

## Design principles

- Nearest-neighbour search is exact, not approximate. Keep it that way, and
  say so in any new docs — silently swapping in an ANN approximation would
  break the "trust what this tool tells you" contract.
- No stubs. Every documented command must actually work end to end.
- Determinism matters: anything with randomness (k-means) must accept a seed
  and produce identical output for identical input and seed.
