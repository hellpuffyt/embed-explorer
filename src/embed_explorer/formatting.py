"""Human-readable table rendering for CLI output."""

from __future__ import annotations

from typing import Any


def render_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a simple fixed-width text table."""
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [fmt_row(headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt_row(row) for row in str_rows)
    return "\n".join(lines)
