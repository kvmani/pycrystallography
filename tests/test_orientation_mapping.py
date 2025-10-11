from __future__ import annotations

from pathlib import Path

import pytest

from pycrystallography.adapters.pymatgen_adapter import StructureLoader
from pycrystallography.analysis.orientation_mapping import map_parent_features_to_child_variants
from pycrystallography.cli.composite import build_orientation_relation, build_phases
from pycrystallography.config import IndexSpec, load_config


pytestmark = pytest.mark.filterwarnings("ignore:Issues encountered while parsing CIF")

REGISTRY = Path(__file__).resolve().parents[1] / "data" / "registry.yaml"


def _load_relation(sample_config: Path):
    cfg = load_config(sample_config)
    loader = StructureLoader.from_yaml(REGISTRY)
    phases = build_phases(cfg, loader)
    relation, variants = build_orientation_relation(cfg, phases, "burgers-zr")
    return cfg, relation, variants


def test_mapping_matches_configuration_pairs(sample_config: Path) -> None:
    cfg, relation, variants = _load_relation(sample_config)
    features = cfg.find_orientation("burgers-zr").parent_directions
    mappings = map_parent_features_to_child_variants(relation, variants, features)

    first = mappings[0]
    assert first.variant.label.endswith("v01")
    child_values = [mapping.child_indices for mapping in first.mappings]
    assert child_values[0] == (0, 0, 1)
    assert child_values[1] == (1, 0, 0)
    assert child_values[2] == (0, 0, 1)
    assert first.mappings[0].child_indices_four == (0, 0, 0, 1)


def test_mapping_generates_four_index_for_hex(sample_config: Path) -> None:
    _cfg, relation, variants = _load_relation(sample_config)
    feature = IndexSpec(kind="direction", indices=(1, 1, 1))
    mappings = map_parent_features_to_child_variants(relation, variants, [feature])
    assert len(mappings) == len(variants)
    for mapping in mappings:
        result = mapping.mappings[0]
        assert result.child_indices_four is not None
        assert len(result.child_indices_four) == 4
        assert any(abs(component) > 0 for component in result.child_indices_four)
