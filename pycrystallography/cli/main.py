"""Typer-based command line interface for pycrystallography."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence

import typer

from .. import __version__
from .._logging import configure_logging
from ..config import IndexSpec, resolve_config
from ..adapters.pymatgen_adapter import StructureLoader
from ..analysis.orientation_mapping import map_parent_features_to_child_variants
from .composite import DEFAULT_REGISTRY, build_orientation_relation, build_phases, run_tem_composite
from .powder import run_powder_xrd

app = typer.Typer(help="Crystallography utilities for composite diffraction")
config_app = typer.Typer(help="Configuration helpers")
composite_app = typer.Typer(help="Composite diffraction workflows")
powder_app = typer.Typer(help="Powder diffraction utilities")
orientation_app = typer.Typer(help="Orientation relationship utilities")

app.add_typer(config_app, name="config")
app.add_typer(composite_app, name="composite")
app.add_typer(powder_app, name="powder")
app.add_typer(orientation_app, name="or")


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
    image_path = run_tem_composite(
        cfg,
        relation_name=relation,
        output=out,
        dry_run=dry_run,
        calculator_slug=calculator,
    )
    typer.echo(f"Composite pattern ready at {image_path}")


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
    orientation_relation, variants = build_orientation_relation(cfg, phases, relation)
    if not variants:
        typer.echo("No child variants were generated for this relation")
        raise typer.Exit(code=1)

    if direction or plane:
        features: List[IndexSpec] = []
        features.extend(_parse_indices_option(direction, "direction"))
        features.extend(_parse_indices_option(plane, "plane"))
    else:
        features = list(relation_cfg.parent_directions)

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
