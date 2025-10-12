from __future__ import annotations

from pathlib import Path

from pycrystallography.io.cif import StructureCache, load_cif_structure


def test_structure_metadata_validation():
    cif = Path(__file__).resolve().parents[1] / "data" / "structureData" / "Zr-Alpha.cif"
    summary = load_cif_structure(cif)
    warnings = summary.validate_metadata({"space_group": summary.space_group})
    assert not warnings
    mismatch = summary.validate_metadata({"space_group": summary.space_group + 1})
    assert mismatch


def test_structure_cache_memoises():
    cif = Path(__file__).resolve().parents[1] / "data" / "structureData" / "Zr-Alpha.cif"
    cache = StructureCache()
    first = cache.get(cif)
    second = cache.get(cif)
    assert first is second
