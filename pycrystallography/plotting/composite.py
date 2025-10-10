"""Composite diffraction plotting helpers."""
from __future__ import annotations

from collections import defaultdict
from typing import Optional, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.models import CompositePattern


def plot_tem_pattern(pattern: CompositePattern, *, ax: Optional[Axes] = None) -> Figure:
    """Plot a composite TEM diffraction pattern."""

    if ax is None:
        fig_obj, axis = plt.subplots(figsize=(8, 5))
        fig = cast(Figure, fig_obj)
        ax = axis
    else:
        fig = cast(Figure, ax.figure)
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(pattern.variant_labels):
        grouped[label].append(idx)
    for label, indices in grouped.items():
        ax.scatter(
            pattern.q_values[indices],
            pattern.intensities[indices],
            label=label,
            s=20,
        )
    ax.set_xlabel(r"$g$ (1/Å)")
    ax.set_ylabel("Relative intensity")
    ax.set_title(pattern.identifier)
    if grouped:
        ax.legend(title="Variant", loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig
