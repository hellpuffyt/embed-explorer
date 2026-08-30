from __future__ import annotations

from embed_explorer.formatting import render_table


def test_render_table_includes_headers_and_rows() -> None:
    out = render_table(["id", "score"], [["a", "0.99"], ["b", "0.5"]])
    lines = out.splitlines()
    assert lines[0].startswith("id")
    assert "a" in lines[2]
    assert "b" in lines[3]


def test_render_table_empty_rows() -> None:
    out = render_table(["id", "score"], [])
    lines = out.splitlines()
    assert len(lines) == 2  # header + separator only


def test_render_table_column_width_adapts_to_content() -> None:
    out = render_table(["id"], [["a-very-long-identifier"]])
    lines = out.splitlines()
    assert "a-very-long-identifier" in lines[2]
