"""Powder diffraction CLI helpers."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import typer

from ..adapters.pymatgen_adapter import DiffractionData, StructureLoader
from ..config import AppConfig
from ..core.models import Phase, PowderPattern
from ..plotting.powder import plot_powder_pattern
from .composite import DEFAULT_REGISTRY


@dataclass(slots=True)
class PowderResult:
    """Bundle of the powder pattern and output artefacts."""

    pattern: PowderPattern
    csv_path: Path
    figure_path: Path


def _build_phase(loader: StructureLoader, cfg) -> Phase:
    structure = loader.get(cfg.structure)
    return Phase(name=cfg.name, structure=structure, metadata=cfg.metadata)


def run_powder_xrd(
    config: AppConfig,
    *,
    phase_name: str,
    two_theta: Tuple[float, float],
    wavelength: float,
    min_intensity: float,
    out: Path,
    dry_run: bool,
) -> PowderResult:
    loader = StructureLoader.from_yaml(DEFAULT_REGISTRY)
    phase_cfg = config.find_phase(phase_name)
    phase = _build_phase(loader, phase_cfg)
    calculator = DiffractionData(wavelength=wavelength, symprec=None)
    two_theta_range = (min(two_theta), max(two_theta))
    two_theta_values, intensities, d_spacings, hkls = calculator.powder_pattern(
        phase.structure,
        two_theta_range=two_theta_range,
        min_intensity=min_intensity,
        wavelength=wavelength,
    )
    metadata = {
        "wavelength": wavelength,
        "two_theta_range": two_theta_range,
        "min_intensity": min_intensity,
    }
    pattern = PowderPattern(
        phase=phase,
        two_theta=two_theta_values,
        intensities=intensities,
        d_spacings=d_spacings,
        hkls=hkls,
        metadata=metadata,
    )
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{phase.name.replace(' ', '_')}_powder.csv"
    fig_path = out / f"{phase.name.replace(' ', '_')}_powder.png"
    if not dry_run:
        table = pattern.to_table()
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(table.dtype.names)
            for row in table:
                writer.writerow([f"{value}" for value in row])
        fig = plot_powder_pattern(pattern)
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        typer.echo(f"Saved powder pattern CSV to {csv_path}")
        typer.echo(f"Saved powder pattern plot to {fig_path}")
    else:
        typer.echo("Dry run: skipping file generation")
    if pattern.intensities.size:
        order = np.argsort(pattern.intensities)[::-1][:5]
        typer.echo("Top reflections:")
        for idx in order:
            hkl = " ".join(str(v) for v in pattern.hkls[idx]) or "-"
            typer.echo(
                f"  {pattern.two_theta[idx]:6.2f}°  I={pattern.intensities[idx]:7.2f}  d={pattern.d_spacings[idx]:.4f} Å  (hkl) {hkl}"
            )
    else:
        typer.echo("No reflections found in the requested range.")
    return PowderResult(pattern=pattern, csv_path=csv_path, figure_path=fig_path)
