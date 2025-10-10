"""Typer-based command line interface for pycrystallography."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .. import __version__
from .._logging import configure_logging
from ..config import resolve_config
from .composite import run_tem_composite

app = typer.Typer(help="Crystallography utilities for composite diffraction")
config_app = typer.Typer(help="Configuration helpers")
composite_app = typer.Typer(help="Composite diffraction workflows")

app.add_typer(config_app, name="config")
app.add_typer(composite_app, name="composite")


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


@app.command("version")
def version() -> None:
    typer.echo(__version__)


def cli() -> None:
    """Run the Typer application."""

    app()


__all__ = ["app", "cli"]
