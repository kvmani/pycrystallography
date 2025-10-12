from __future__ import annotations

from pathlib import Path

from pycrystallography.adapters.orix_adapter import VariantGenerator
from pycrystallography.adapters.pymatgen_adapter import StructureLoader
from pycrystallography.cli.composite import build_orientation_relation, build_phases
from pycrystallography.config import load_config


def test_variant_generation_deterministic(sample_config):
    cfg = load_config(sample_config)
    registry_path = Path(__file__).resolve().parents[1] / "data" / "registry.yaml"
    loader = StructureLoader.from_yaml(registry_path)
    phases = build_phases(cfg, loader)
    relation, variants, _spec = build_orientation_relation(cfg, phases, "burgers-zr")
    relation_cfg = cfg.find_orientation("burgers-zr")
    generator = VariantGenerator.from_space_groups(
        cfg.find_phase(relation_cfg.parent_phase).space_group,
        cfg.find_phase(relation_cfg.child_phase).space_group,
    )
    first = generator.generate(relation)
    second = generator.generate(relation)
    assert [v.label for v in first] == [v.label for v in second] == [v.label for v in variants]
    assert len({v.label for v in first}) == len(first)
