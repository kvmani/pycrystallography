"""Typer-based command line interface for pycrystallography."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import typer

from .. import __version__
from .._logging import configure_logging
from ..config import IndexSpec, resolve_config
from ..adapters.pymatgen_adapter import StructureLoader
from ..analysis.orientation_mapping import map_parent_features_to_child_variants
from ..core.models import StereographicPattern
from ..core.variant_manager import MarkerPalette, VariantManager
from ..plotting import PlotSettings, StereographicFigure
from .composite import DEFAULT_REGISTRY, build_orientation_relation, build_phases, run_tem_composite
from .powder import run_powder_xrd

app = typer.Typer(help="Crystallography utilities for composite diffraction")
config_app = typer.Typer(help="Configuration helpers")
composite_app = typer.Typer(help="Composite diffraction workflows")
powder_app = typer.Typer(help="Powder diffraction utilities")
orientation_app = typer.Typer(help="Orientation relationship utilities")
plot_app = typer.Typer(help="Plotting helpers")
report_app = typer.Typer(help="Reporting helpers")

app.add_typer(config_app, name="config")
app.add_typer(composite_app, name="composite")
app.add_typer(powder_app, name="powder")
app.add_typer(orientation_app, name="or")
app.add_typer(plot_app, name="plot")
app.add_typer(report_app, name="report")


def _parse_indices_option(values: Sequence[str], kind: str) -> List[IndexSpec]:
    specs: List[IndexSpec] = []
    for raw in values:
        tokens = [token for token in re.split(r"[\s,]+", raw.strip()) if token]
        if not tokens:
            raise typer.BadParameter("Index specifications must not be empty")
        try:
            components = tuple(float(token) for token in tokens)
        except ValueError as exc:
            raise typer.BadParameter(f"Could not parse indices from '{raw}'") from exc
        specs.append(IndexSpec(kind=kind, indices=components))
    return specs


def _format_indices(kind: str, values: Sequence[int | float]) -> str:
    def _format_value(value: int | float) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        if isinstance(value, (int, float)):
            return f"{value:g}"
        return str(value)

    formatted = " ".join(_format_value(component) for component in values)
    if kind == "plane":
        return f"({formatted})"
    return f"[{formatted}]"


def _format_spec(spec: IndexSpec) -> str:
    descriptor = _format_indices(spec.kind, spec.indices)
    if spec.label:
        return f"{spec.label} {descriptor}"
    return descriptor


def _load_plot_settings(config) -> PlotSettings:
    config_dir = Path(getattr(config, "_config_dir", Path.cwd()))
    settings_path = config_dir / "configs" / "plot_settings.yaml"
    if not settings_path.exists():
        settings_path = Path(__file__).resolve().parents[2] / "configs" / "plot_settings.yaml"
    return PlotSettings.from_yaml(settings_path)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging")) -> None:
    level = "DEBUG" if verbose else "INFO"
    configure_logging(level=level)


@config_app.command("validate")
def validate_config(
    config: Optional[Path] = typer.Option(None, "--config", exists=True, help="Configuration file"),
    print_config: bool = typer.Option(False, "--print", help="Print the validated configuration"),
) -> None:
    cfg = resolve_config(config_path=config)
    if print_config:
        typer.echo(cfg.model_dump_json(indent=2))
    typer.echo("Configuration is valid ✅")


@composite_app.command("tem")
def composite_tem(
    relation: str = typer.Option(..., "--relation", help="Orientation relation name"),
    config: Optional[Path] = typer.Option(None, "--config", exists=True),
    out: Path = typer.Option(Path("output"), "--out", help="Output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip writing files"),
    calculator: Optional[str] = typer.Option(None, "--calculator", help="Override calculator slug"),
) -> None:
    cfg = resolve_config(config_path=config)
    artifacts = run_tem_composite(
        cfg,
        relation_name=relation,
        output=out,
        dry_run=dry_run,
        calculator_slug=calculator,
    )
    if not dry_run:
        typer.echo(f"Composite pattern ready at {artifacts.image_path}")


@powder_app.command("xrd")
def powder_xrd(
    phase: str = typer.Option(..., "--phase", help="Phase name from the configuration"),
    config: Optional[Path] = typer.Option(None, "--config", exists=True),
    two_theta_min: float = typer.Option(5.0, "--two-theta-min", help="Lower 2θ bound"),
    two_theta_max: float = typer.Option(90.0, "--two-theta-max", help="Upper 2θ bound"),
    wavelength: float = typer.Option(1.5406, "--wavelength", help="X-ray wavelength in Å"),
    min_intensity: float = typer.Option(1.0, "--min-intensity", help="Discard peaks below this intensity"),
    out: Path = typer.Option(Path("output"), "--out", help="Output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip writing files"),
) -> None:
    cfg = resolve_config(config_path=config)
    result = run_powder_xrd(
        cfg,
        phase_name=phase,
        two_theta=(two_theta_min, two_theta_max),
        wavelength=wavelength,
        min_intensity=min_intensity,
        out=out,
        dry_run=dry_run,
    )
    typer.echo(
        f"Powder pattern computed for {phase} over {two_theta_min:.1f}–{two_theta_max:.1f}°"
    )
    if not dry_run:
        typer.echo(f"Artifacts saved to: {result.csv_path.parent}")


@plot_app.command("composite")
def plot_composite(
    relation: str = typer.Option(..., "--relation", help="Orientation relation name"),
    config: Optional[Path] = typer.Option(None, "--config", exists=True),
    out: Path = typer.Option(Path("output"), "--out", help="Output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip writing files"),
    calculator: Optional[str] = typer.Option(None, "--calculator", help="Override calculator slug"),
) -> None:
    cfg = resolve_config(config_path=config)
    artifacts = run_tem_composite(
        cfg,
        relation_name=relation,
        output=out,
        dry_run=dry_run,
        calculator_slug=calculator,
    )
    if dry_run:
        typer.echo("Dry run: no figure written")
    else:
        typer.echo(f"Interactive figure saved to {artifacts.image_path}")


@plot_app.command("stereographic")
def plot_stereographic(
    relation: str = typer.Option(..., "--relation", help="Orientation relation name"),
    config: Optional[Path] = typer.Option(None, "--config", exists=True),
    show: bool = typer.Option(True, "--show/--no-show", help="Display the interactive figure"),
    backend: Optional[str] = typer.Option(None, "--backend", help="Matplotlib backend override"),
) -> None:
    cfg = resolve_config(config_path=config)
    loader = StructureLoader.from_yaml(DEFAULT_REGISTRY)
    phases = build_phases(cfg, loader)
    relation_obj, variants, spec = build_orientation_relation(cfg, phases, relation)
    relation_cfg = cfg.find_orientation(relation)
    features: List[IndexSpec] = list(relation_cfg.parent_directions)
    if not features and spec is not None:
        features = [
            IndexSpec(kind="direction", indices=tuple(float(v) for v in spec.uvw_parent), label="parent ⟨uvw⟩"),
            IndexSpec(kind="plane", indices=tuple(float(v) for v in spec.hkl_parent), label="parent (hkl)"),
        ]
    if not features:
        features = [
            IndexSpec(kind="direction", indices=(1, 0, 0), label="[100]"),
            IndexSpec(kind="direction", indices=(0, 1, 0), label="[010]"),
            IndexSpec(kind="direction", indices=(0, 0, 1), label="[001]"),
        ]
    variant_mappings = map_parent_features_to_child_variants(relation_obj, variants, features)
    polar: List[tuple[float, float]] = []
    variant_labels: List[str] = []
    hemispheres: List[bool] = []
    labels: List[str] = []
    for mapping in variant_mappings:
        for feature in mapping.mappings:
            vector = feature.child_cartesian
            hemisphere = bool(vector[2] >= 0)
            hemispheres.append(hemisphere)
            z_clamped = max(min(abs(float(vector[2])), 1.0), -1.0)
            theta = math.acos(z_clamped)
            phi = math.atan2(float(vector[1]), float(vector[0]))
            polar.append((theta, phi))
            variant_labels.append(mapping.variant.label)
            labels.append(feature.spec.label or _format_spec(feature.spec))
    if not polar:
        typer.echo("No stereographic poles were generated", err=True)
        raise typer.Exit(code=1)
    pattern = StereographicPattern(
        identifier=f"{relation_obj.name} stereographic",
        variants=tuple(variants),
        polar_angles=np.array(polar, dtype=float),
        variant_labels=np.array(variant_labels, dtype="U32"),
        hemispheres=np.array(hemispheres, dtype=bool),
        labels=tuple(labels),
        metadata={"features": [feature.model_dump() for feature in features]},
    )
    settings = _load_plot_settings(cfg)
    palette = MarkerPalette(
        shapes=settings.markers.get("shapes", ("o",)),
        colors=settings.markers.get("colors", ("#1f77b4",)),
        fallback_shape=settings.markers.get("fallback_shape", "o"),
        fallback_color=settings.markers.get("fallback_color", "#444444"),
    )
    manager = VariantManager(pattern.variants, palette=palette)
    figure = StereographicFigure(pattern, manager, settings=settings, backend=backend)
    typer.echo(f"Prepared stereographic pattern with {pattern.polar_angles.shape[0]} poles")
    if show:
        figure.show()
    else:
        typer.echo("Display suppressed (--no-show)")


@report_app.command("phase")
def report_phase(
    relation: str = typer.Option(..., "--relation", help="Orientation relation name"),
    config: Optional[Path] = typer.Option(None, "--config", exists=True),
    out: Path = typer.Option(Path("output"), "--out", help="Output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip writing files"),
    calculator: Optional[str] = typer.Option(None, "--calculator", help="Override calculator slug"),
) -> None:
    cfg = resolve_config(config_path=config)
    artifacts = run_tem_composite(
        cfg,
        relation_name=relation,
        output=out,
        dry_run=dry_run,
        calculator_slug=calculator,
    )
    if dry_run:
        typer.echo("Dry run: report generation skipped")
    elif artifacts.report_path:
        typer.echo(f"HTML report available at {artifacts.report_path}")
    else:
        typer.echo("No HTML report generated (missing OR metadata)")


@orientation_app.command("map")
def map_or_features(
    relation: str = typer.Option(..., "--relation", help="Orientation relation name"),
    config: Optional[Path] = typer.Option(None, "--config", exists=True),
    direction: List[str] = typer.Option([], "--direction", help="Parent direction indices"),
    plane: List[str] = typer.Option([], "--plane", help="Parent plane indices"),
) -> None:
    cfg = resolve_config(config_path=config)
    relation_cfg = cfg.find_orientation(relation)
    loader = StructureLoader.from_yaml(DEFAULT_REGISTRY)
    phases = build_phases(cfg, loader)
    orientation_relation, variants, spec = build_orientation_relation(cfg, phases, relation)
    if not variants:
        typer.echo("No child variants were generated for this relation")
        raise typer.Exit(code=1)

    if direction or plane:
        features: List[IndexSpec] = []
        features.extend(_parse_indices_option(direction, "direction"))
        features.extend(_parse_indices_option(plane, "plane"))
    else:
        features = list(relation_cfg.parent_directions)
        if not features and spec is not None:
            features = [
                IndexSpec(kind="direction", indices=spec.uvw_parent),
                IndexSpec(kind="plane", indices=spec.hkl_parent),
            ]

    if not features:
        typer.echo("No parent features supplied to map")
        raise typer.Exit(code=1)

    mappings = map_parent_features_to_child_variants(orientation_relation, variants, features)

    typer.echo(f"Orientation relation: {orientation_relation.name}")
    typer.echo(f"Parent phase: {orientation_relation.parent_phase.name}")
    typer.echo(f"Child phase: {orientation_relation.child_phase.name}")
    typer.echo("")
    for variant_mapping in mappings:
        typer.echo(f"Variant {variant_mapping.variant.label}")
        for feature in variant_mapping.mappings:
            parent_desc = _format_spec(feature.spec)
            child_desc = _format_indices(feature.spec.kind, feature.child_indices)
            if feature.child_indices_four is not None:
                four_desc = _format_indices(feature.spec.kind, feature.child_indices_four)
                child_desc = f"{child_desc} (four-index {four_desc})"
            typer.echo(
                f"  {feature.spec.kind.capitalize():<9} {parent_desc} -> {child_desc}"
            )
        typer.echo("")


@app.command("version")
def version() -> None:
    typer.echo(__version__)


def cli() -> None:
    """Run the Typer application."""

    app()


__all__ = ["app", "cli"]
