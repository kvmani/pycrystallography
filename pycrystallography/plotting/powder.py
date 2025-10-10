"""Plotting helpers for powder diffraction patterns."""
from __future__ import annotations

from typing import Optional, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.models import PowderPattern


def plot_powder_pattern(pattern: PowderPattern, *, ax: Optional[Axes] = None) -> Figure:
    """Plot a powder XRD pattern as a line chart."""

    if ax is None:
        fig_obj, axis = plt.subplots(figsize=(8, 5))
        fig = cast(Figure, fig_obj)
        ax = axis
    else:
        fig = cast(Figure, ax.figure)
    ax.plot(pattern.two_theta, pattern.intensities, color="#1f77b4", lw=1.5)
    ax.set_xlabel(r"$2\theta$ (degrees)")
    ax.set_ylabel("Relative intensity")
    ax.set_title(f"Powder XRD: {pattern.phase.name}")
    if pattern.two_theta.size:
        ax.set_xlim(float(pattern.two_theta.min()), float(pattern.two_theta.max()))
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig
