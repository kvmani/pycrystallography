from __future__ import annotations

from pathlib import Path

from pycrystallography.adapters.orix_adapter import OrientationFactory, VariantGenerator
from pycrystallography.adapters.pymatgen_adapter import StructureLoader
from pycrystallography.config import load_config
from pycrystallography.core.models import OrientationRelation, Phase
from pycrystallography.core.indexing import indices_to_cartesian


def test_variant_generation_deterministic(sample_config):
    cfg = load_config(sample_config)
    registry_path = Path(__file__).resolve().parents[1] / "data" / "registry.yaml"
    loader = StructureLoader.from_yaml(registry_path)
    phases = {
        phase_cfg.name: Phase(
            name=phase_cfg.name,
            structure=loader.get(phase_cfg.structure),
            metadata=phase_cfg.metadata,
        )
        for phase_cfg in cfg.phases
    }
    relation_cfg = cfg.find_orientation("burgers-zr")
    factory = OrientationFactory()
    parent_structure = phases[relation_cfg.parent_phase].structure
    child_structure = phases[relation_cfg.child_phase].structure
    parent_vectors = [
        indices_to_cartesian(parent_structure, kind=spec.kind, indices=spec.indices)
        for spec in relation_cfg.parent_directions
    ]
    child_vectors = [
        indices_to_cartesian(child_structure, kind=spec.kind, indices=spec.indices)
        for spec in relation_cfg.child_directions
    ]
    orientation = factory.from_direction_pairs(parent_vectors, child_vectors)
    relation = OrientationRelation(
        name=relation_cfg.name,
        parent_phase=phases[relation_cfg.parent_phase],
        child_phase=phases[relation_cfg.child_phase],
        orientation=orientation,
    )
    generator = VariantGenerator.from_space_groups(
        cfg.find_phase(relation_cfg.parent_phase).space_group,
        cfg.find_phase(relation_cfg.child_phase).space_group,
    )
    first = generator.generate(relation)
    second = generator.generate(relation)
    assert [v.label for v in first] == [v.label for v in second]
    assert len({v.label for v in first}) == len(first)
