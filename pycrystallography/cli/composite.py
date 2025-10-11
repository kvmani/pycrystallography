"""Composite diffraction CLI helpers."""
from __future__ import annotations

from pathlib import Path
import csv
from typing import List

import typer
from orix.quaternion.orientation import Orientation

from ..adapters.orix_adapter import OrientationFactory, VariantGenerator
from ..adapters.pymatgen_adapter import StructureLoader
from ..config import AppConfig
from ..core.indexing import indices_to_cartesian
from ..core.models import OrientationRelation, Phase, Variant
from ..plotting.composite import plot_tem_pattern
DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "data" / "registry.yaml"


def _build_phase(loader: StructureLoader, cfg) -> Phase:
    structure = loader.get(cfg.structure)
    return Phase(name=cfg.name, structure=structure, metadata=cfg.metadata)


def build_phases(config: AppConfig, loader: StructureLoader | None = None) -> dict[str, Phase]:
    """Materialise all phases defined in the configuration."""

    loader = loader or StructureLoader.from_yaml(DEFAULT_REGISTRY)
    return {cfg.name: _build_phase(loader, cfg) for cfg in config.phases}


def build_orientation_relation(
    config: AppConfig,
    phases: dict[str, Phase],
    relation_name: str,
) -> tuple[OrientationRelation, List[Variant]]:
    relation_cfg = config.find_orientation(relation_name)
    parent_phase = phases[relation_cfg.parent_phase]
    child_phase = phases[relation_cfg.child_phase]
    factory = OrientationFactory()
    parent_vectors = [
        indices_to_cartesian(
            parent_phase.structure, kind=spec.kind, indices=spec.indices
        )
        for spec in relation_cfg.parent_directions
    ]
    child_vectors = [
        indices_to_cartesian(
            child_phase.structure, kind=spec.kind, indices=spec.indices
        )
        for spec in relation_cfg.child_directions
    ]
    orientation = factory.from_direction_pairs(parent_vectors, child_vectors)
    orientation_relation = OrientationRelation(
        name=relation_cfg.name,
        parent_phase=parent_phase,
        child_phase=child_phase,
        orientation=orientation,
    )
    generator = VariantGenerator.from_space_groups(
        config.find_phase(relation_cfg.parent_phase).space_group,
        config.find_phase(relation_cfg.child_phase).space_group,
    )
    variants = list(generator.generate(orientation_relation))
    return orientation_relation, variants


def run_tem_composite(
    config: AppConfig,
    *,
    relation_name: str,
    output: Path,
    dry_run: bool,
    calculator_slug: str | None = None,
) -> Path:
    loader = StructureLoader.from_yaml(DEFAULT_REGISTRY)
    phases = build_phases(config, loader)
    orientation_relation, variants = build_orientation_relation(config, phases, relation_name)

    parent_relation = OrientationRelation(
        name=f"{orientation_relation.name}-parent",
        parent_phase=orientation_relation.parent_phase,
        child_phase=orientation_relation.parent_phase,
        orientation=Orientation.identity(),
    )
    parent_variants = list(
        VariantGenerator.from_space_groups(
            config.find_phase(orientation_relation.parent_phase.name).space_group,
            config.find_phase(orientation_relation.parent_phase.name).space_group,
        ).generate(parent_relation)
    )
    all_variants = tuple(parent_variants + variants)

    from ..analysis.registry import get_calculator

    calc = get_calculator(calculator_slug or config.tem.calculator)
    pattern = calc.compute(
        variants=all_variants,
        metadata={
            "zone_axis": config.tem.zone_axis,
            "identifier": relation_name,
            "voltage": config.tem.voltage,
            "camera_length": config.tem.camera_length,
            "intensity_threshold": config.tem.intensity_threshold,
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{relation_name}_pattern.csv"
    img_path = output / f"{relation_name}_pattern.png"
    if not dry_run:
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["q", "intensity", "variant"])
            for q_value, intensity, label in zip(
                pattern.q_values, pattern.intensities, pattern.variant_labels
            ):
                writer.writerow([f"{q_value:.6f}", f"{intensity:.6f}", label])
        fig = plot_tem_pattern(pattern)
        fig.savefig(img_path, dpi=300, bbox_inches="tight")
        typer.echo(f"Saved pattern CSV to {csv_path}")
        typer.echo(f"Saved plot to {img_path}")
    else:
        typer.echo("Dry run: skipping file generation")
    return img_path
