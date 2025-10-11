from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize("expected_rows", [6])
def test_feature_comparison_row_count(expected_rows: int) -> None:
    doc = Path(__file__).resolve().parents[1] / "docs" / "feature_comparison_main_vs_advanced.md"
    lines = doc.read_text().splitlines()
    rows = []
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().split("|")[1:-1]]
        if not cells or cells[0] in {"#", "-"}:
            continue
        if cells[0].isdigit():
            rows.append(cells)
    assert len(rows) == expected_rows
