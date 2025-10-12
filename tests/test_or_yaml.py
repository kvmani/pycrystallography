from __future__ import annotations

from pathlib import Path

import pytest

from pycrystallography.adapters.pymatgen_adapter import StructureLoader
from pycrystallography.core.models import Phase
from pycrystallography.io.or_yaml import load_orientation_document, load_orientation_library


EXAMPLE_OR = Path(__file__).resolve().parents[1] / "examples" / "or_zr.yaml"
REGISTRY = Path(__file__).resolve().parents[1] / "data" / "registry.yaml"


def test_orientation_document_loads_and_validates():
    document = load_orientation_document(EXAMPLE_OR)
    assert document.phase == "alpha-zr"
    assert document.orientation_relations[0].name == "burgers-zr"


def test_orientation_library_generates_variants():
    loader = StructureLoader.from_yaml(REGISTRY)
    parent = Phase(name="beta-zr", structure=loader.get("zr_beta"))
    library = load_orientation_library([EXAMPLE_OR])
    relation, variants, spec = library.build_orientation(
        "burgers-zr", parent_phase=parent, child_phase_name="alpha-zr"
    )
    assert relation.child_phase.name == "alpha-zr"
    assert len(variants) >= 2
    assert spec.variants.defaults.visible is True


def test_invalid_orientation_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
phase: foo
orientation_relations:
  - name: bad
    hkl_parent: [1, 0, 0]
    uvw_parent: [1, 0]
    hkl_product: [1, 0, 0]
    uvw_product: [1, 0, 0]
"""
    )
    with pytest.raises(ValueError):
        load_orientation_document(bad)
