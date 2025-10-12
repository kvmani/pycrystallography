"""Composite diffraction CLI helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import List

import typer
from orix.quaternion.orientation import Orientation

from ..adapters.orix_adapter import OrientationFactory, VariantGenerator
from ..adapters.pymatgen_adapter import StructureLoader
from ..analysis.reports.html import HtmlReportBuilder
from ..config import AppConfig
from ..core.indexing import indices_to_cartesian
from ..core.models import OrientationRelation, Phase, Variant
from ..core.variant_manager import MarkerPalette, VariantManager
from ..io.or_yaml import OrientationRelationSpec, load_orientation_library
from ..plotting import CrystallographicFigure, PlotSettings
DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "data" / "registry.yaml"


def _build_phase(loader: StructureLoader, cfg) -> Phase:
    structure = loader.get(cfg.structure)
    return Phase(name=cfg.name, structure=structure, metadata=cfg.metadata)


def _config_dir(config: AppConfig) -> Path:
    return getattr(config, "_config_dir", Path.cwd())


@dataclass(slots=True)
class CompositeArtifacts:
    csv_path: Path
    image_path: Path
    report_path: Path | None


def build_phases(config: AppConfig, loader: StructureLoader | None = None) -> dict[str, Phase]:
    """Materialise all phases defined in the configuration."""

    loader = loader or StructureLoader.from_yaml(DEFAULT_REGISTRY)
    return {cfg.name: _build_phase(loader, cfg) for cfg in config.phases}


def build_orientation_relation(
    config: AppConfig,
    phases: dict[str, Phase],
    relation_name: str,
) -> tuple[OrientationRelation, List[Variant], OrientationRelationSpec | None]:
    relation_cfg = config.find_orientation(relation_name)
    parent_phase = phases[relation_cfg.parent_phase]
    child_phase = phases[relation_cfg.child_phase]
    if relation_cfg.or_document:
        doc_path = relation_cfg.or_document
        if not doc_path.is_absolute():
            doc_path = (_config_dir(config) / doc_path).resolve()
        library = load_orientation_library([doc_path])
        relation, variants, spec = library.build_orientation(
            relation_cfg.or_name or relation_cfg.name,
            parent_phase=parent_phase,
            child_phase_name=child_phase.name,
        )
        return relation, list(variants), spec
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
    parent_planes = [spec for spec in relation_cfg.parent_directions if spec.kind == "plane"]
    parent_dirs = [spec for spec in relation_cfg.parent_directions if spec.kind != "plane"]
    child_planes = [spec for spec in relation_cfg.child_directions if spec.kind == "plane"]
    child_dirs = [spec for spec in relation_cfg.child_directions if spec.kind != "plane"]
    spec = OrientationRelationSpec(
        name=relation_cfg.name,
        hkl_parent=tuple(int(v) for v in (parent_planes[0].indices if parent_planes else parent_dirs[0].indices)),
        uvw_parent=tuple(int(v) for v in parent_dirs[0].indices),
        hkl_product=tuple(int(v) for v in (child_planes[0].indices if child_planes else child_dirs[0].indices)),
        uvw_product=tuple(int(v) for v in child_dirs[0].indices),
    )
    return orientation_relation, variants, spec


def run_tem_composite(
    config: AppConfig,
    *,
    relation_name: str,
    output: Path,
    dry_run: bool,
    calculator_slug: str | None = None,
) -> CompositeArtifacts:
    loader = StructureLoader.from_yaml(DEFAULT_REGISTRY)
    phases = build_phases(config, loader)
    orientation_relation, variants, spec = build_orientation_relation(
        config, phases, relation_name
    )

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
    settings_path = (_config_dir(config) / "configs" / "plot_settings.yaml")
    if not settings_path.exists():
        settings_path = Path(__file__).resolve().parents[2] / "configs" / "plot_settings.yaml"
    plot_settings = PlotSettings.from_yaml(settings_path)
    palette = MarkerPalette(
        shapes=plot_settings.markers.get("shapes", ("o",)),
        colors=plot_settings.markers.get("colors", ("#1f77b4",)),
        fallback_shape=plot_settings.markers.get("fallback_shape", "o"),
        fallback_color=plot_settings.markers.get("fallback_color", "#444444"),
    )
    default_visibility = None
    if spec and spec.plot and spec.plot.visibility:
        default_visibility = spec.plot.visibility
    manager = VariantManager(all_variants, default_visibility=default_visibility, palette=palette)
    figure = CrystallographicFigure(pattern, manager, settings=plot_settings)
    report_path: Path | None = None
    if not dry_run:
        table = pattern.to_table()
        headers = list(table.dtype.names)
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            for row in table:
                writer.writerow([row[name] for name in headers])
        figure.save(img_path)
        typer.echo(f"Saved pattern CSV to {csv_path}")
        typer.echo(f"Saved plot to {img_path}")
        if spec is not None:
            builder = HtmlReportBuilder(output)
            report = builder.build(
                phase=orientation_relation.child_phase,
                relation=orientation_relation,
                variants=list(manager),
                orientation_spec=spec,
                figure=figure,
                configuration=config.model_dump(),
            )
            typer.echo(f"Saved HTML report to {report.html_path}")
            report_path = report.html_path
    else:
        typer.echo("Dry run: skipping file generation")
    return CompositeArtifacts(csv_path=csv_path, image_path=img_path, report_path=report_path)
