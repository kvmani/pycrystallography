"""Generate a simulated orientation distribution function (ODF).

This script creates a CSV file containing Euler angles (phi1, phi, phi2)
and their corresponding volume fractions.  The orientation range for each
angle is derived from the symmetry limits provided by
:class:`pycrystallography.core.orientedLattice.OrientedLattice`.

Example
-------
    params = {
        "crystal_symmetry": "cubic",
        "odf_grid_spacing": 5,
        "angle_units": "degree",
        "output_file": "simulated_odf.csv",
    }
    generate_simulated_odf(params)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, Iterable

# Allow running this script directly without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pycrystallography.core.orientedLattice import OrientedLattice


_SYMMETRY_FACTORY = {
    "cubic": lambda: OrientedLattice.cubic(1),
    "tetragonal": lambda: OrientedLattice.tetragonal(1, 1),
    "orthorhombic": lambda: OrientedLattice.orthorhombic(1, 1, 1),
    "monoclinic": lambda: OrientedLattice.monoclinic(1, 1, 1, 90),
    "hexagonal": lambda: OrientedLattice.hexagonal(1, 1.59),
    "rhombohedral": lambda: OrientedLattice.rhombohedral(1, 90),
    "triclinic": lambda: OrientedLattice.from_parameters(1, 1, 1, 90, 90, 90),
}


def _get_euler_limits(symmetry: str) -> Iterable[float]:
    """Return Euler angle limits in radians for the given symmetry."""
    key = symmetry.lower()
    if key not in _SYMMETRY_FACTORY:
        raise ValueError(f"Unsupported crystal symmetry: {symmetry}")
    lattice = _SYMMETRY_FACTORY[key]()
    return lattice._EulerLimits


def generate_simulated_odf(params: Dict[str, object]) -> str:
    """Generate ODF grid with random weights and save as CSV.

    Parameters
    ----------
    params : dict
        Dictionary with the following optional keys::

            {
                "crystal_symmetry": "cubic",        # one of cubic, tetragonal...
                "odf_grid_spacing": 5,              # spacing of Euler grid
                "angle_units": "degree",            # 'degree' or 'radian'
                "output_file": "simulated_odf.csv"  # file name for CSV
            }

    Returns
    -------
    str
        Path to the generated CSV file.
    """
    symmetry = params.get("crystal_symmetry", "cubic")
    spacing = float(params.get("odf_grid_spacing", 5))
    units = params.get("angle_units", "degree").lower()
    output_file = params.get("output_file", "simulated_odf.csv")

    limits_rad = _get_euler_limits(symmetry)
    if units.startswith("deg"):
        limits = np.degrees(limits_rad)
        step = spacing
    else:
        limits = limits_rad
        step = spacing

    phi1 = np.arange(0, limits[0] + 1e-6, step)
    phi = np.arange(0, limits[1] + 1e-6, step)
    phi2 = np.arange(0, limits[2] + 1e-6, step)

    grid = []
    for a in phi1:
        for b in phi:
            for c in phi2:
                grid.append([a, b, c])
    grid = np.array(grid)
    weights = np.random.uniform(0, 100, size=len(grid))
    weights /= weights.sum()

    df = pd.DataFrame(
        np.column_stack([grid, weights]),
        columns=["phi1", "phi", "phi2", "volume fraction"],
    )
    df.to_csv(output_file, index=False)
    return output_file


if __name__ == "__main__":
    example = {
        "crystal_symmetry": "cubic",
        "odf_grid_spacing": 5,
        "angle_units": "degree",
        "output_file": "simulated_odf.csv",
    }
    path = generate_simulated_odf(example)
    print(f"ODF written to {path}")
