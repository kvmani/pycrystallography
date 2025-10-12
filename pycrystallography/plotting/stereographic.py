"""Interactive stereographic projection figure."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.widgets import CheckButtons

from ..core.models import StereographicPattern
from ..core.variant_manager import VariantManager, VariantState
from .annotation import AnnotationConfig, AnnotationManager
from .settings import PlotSettings


@dataclass(slots=True)
class StereographicPayload:
    """Payload describing a selected stereographic pole."""

    variant_label: str
    x: float
    y: float
    text: str
    hemisphere: bool
    index: int


class StereographicFigure:
    """Interactive figure for stereographic pole plots."""

    def __init__(
        self,
        pattern: StereographicPattern,
        variant_manager: VariantManager,
        *,
        settings: Optional[PlotSettings] = None,
        backend: Optional[str] = None,
    ) -> None:
        self.pattern = pattern
        self.variant_manager = variant_manager
        if backend:
            plt.switch_backend(backend)
        self.settings = settings or PlotSettings.from_mapping({})
        self.figure, self.ax = self._create_figure()
        annotation_cfg = AnnotationConfig(
            max_labels=int(self.settings.annotations.get("max_labels", 30)),
            min_distance=float(self.settings.annotations.get("min_distance", 10.0)),
            offset_radius=float(self.settings.annotations.get("offset_radius", 10.0)),
            polar_steps=int(self.settings.annotations.get("polar_steps", 24)),
            font_size=float(self.settings.annotations.get("font_size", 9)),
            box_facecolor=str(self.settings.annotations.get("box_facecolor", "#fffff8")),
            box_edgecolor=str(self.settings.annotations.get("box_edgecolor", "#666666")),
            box_alpha=float(self.settings.annotations.get("box_alpha", 0.9)),
        )
        self._annotation_manager = AnnotationManager(self.ax, annotation_cfg)
        self._hover_threshold_sq = annotation_cfg.min_distance ** 2
        self._scatter_by_label: Dict[str, Dict[str, Optional[PathCollection]]] = {}
        self._point_records: Dict[str, Dict[str, object]] = {}
        self._legend = None
        self._check_buttons: Optional[CheckButtons] = None
        self._prepare_points()
        self._init_widgets()
        self._connect_events()

    def _create_figure(self) -> Tuple[Figure, Axes]:
        figure_kwargs = dict(self.settings.figure)
        figsize = figure_kwargs.pop("figsize", (6.0, 6.0))
        dpi = figure_kwargs.pop("dpi", 120)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        if "facecolor" in figure_kwargs:
            fig.patch.set_facecolor(figure_kwargs["facecolor"])
        ax.set_aspect("equal")
        ax.set_title(self.pattern.identifier)
        ax.set_xlabel(r"$X$")
        ax.set_ylabel(r"$Y$")
        ax.set_xticks([])
        ax.set_yticks([])
        circle = Circle((0.0, 0.0), radius=1.0, fill=False, linewidth=1.2, edgecolor="#1f2933")
        ax.add_patch(circle)
        ax.axhline(0.0, color="#9aa5b1", linestyle="--", linewidth=0.8)
        ax.axvline(0.0, color="#9aa5b1", linestyle="--", linewidth=0.8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        return fig, ax

    def _prepare_points(self) -> None:
        coords = self.pattern.cartesian_coordinates()
        scatter_kwargs = dict(self.settings.scatter)
        size = float(scatter_kwargs.get("size", 40))
        alpha = float(scatter_kwargs.get("alpha", 0.9))
        linewidth = float(scatter_kwargs.get("linewidth", 0.6))
        edgecolor = scatter_kwargs.get("edgecolor", "#222222")
        legend_handles: list[PathCollection] = []
        legend_labels: list[str] = []
        for state in self.variant_manager:
            indices = np.nonzero(self.pattern.variant_labels == state.variant.label)[0]
            x_values = coords[indices, 0] if len(indices) else np.empty(0)
            y_values = coords[indices, 1] if len(indices) else np.empty(0)
            hemispheres = self.pattern.hemispheres[indices] if len(indices) else np.empty(0, dtype=bool)
            labels = [self.pattern.labels[idx] for idx in indices]
            artists: Dict[str, Optional[PathCollection]] = {"upper": None, "lower": None}
            record = {
                "state": state,
                "x": np.array(x_values, dtype=float),
                "y": np.array(y_values, dtype=float),
                "labels": labels,
                "hemispheres": np.array(hemispheres, dtype=bool),
            }
            legend_handle: Optional[PathCollection] = None
            if len(indices):
                upper_mask = record["hemispheres"]
                lower_mask = ~upper_mask
                if np.any(upper_mask):
                    artist_upper = self.ax.scatter(
                        record["x"][upper_mask],
                        record["y"][upper_mask],
                        marker=state.style.marker,
                        s=size,
                        alpha=alpha,
                        linewidths=linewidth,
                        edgecolors=edgecolor,
                        facecolors=state.style.color,
                        visible=state.visible,
                        label=state.variant.label,
                    )
                    artists["upper"] = artist_upper
                    legend_handle = legend_handle or artist_upper
                if np.any(lower_mask):
                    artist_lower = self.ax.scatter(
                        record["x"][lower_mask],
                        record["y"][lower_mask],
                        marker=state.style.marker,
                        s=size,
                        alpha=1.0,
                        linewidths=linewidth,
                        edgecolors=state.style.color,
                        facecolors="none",
                        visible=state.visible,
                        label=state.variant.label if legend_handle is None else None,
                    )
                    artists["lower"] = artist_lower
                    legend_handle = legend_handle or artist_lower
            self._scatter_by_label[state.variant.label] = artists
            self._point_records[state.variant.label] = record
            if legend_handle is not None:
                legend_handles.append(legend_handle)
                legend_labels.append(state.variant.label)
        if legend_handles:
            self._legend = self.ax.legend(legend_handles, legend_labels, title="Variant", loc="upper right")

    def _init_widgets(self) -> None:
        widget_cfg = self.settings.widgets
        anchor = widget_cfg.get("anchor", (0.78, 0.55))
        width = float(widget_cfg.get("width", 0.2))
        height = float(widget_cfg.get("height", 0.3))
        labels = [state.variant.label for state in self.variant_manager]
        states = [state.visible for state in self.variant_manager]
        checkbox_ax = self.figure.add_axes([anchor[0], anchor[1], width, height])
        self._check_buttons = CheckButtons(checkbox_ax, labels, states)
        self._check_buttons.on_clicked(self._handle_toggle)
        checkbox_ax.set_title("Variants")

    def _connect_events(self) -> None:
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)

    def _nearest_point(self, event) -> Optional[StereographicPayload]:  # type: ignore[no-untyped-def]
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        event_xy = np.array([event.x, event.y])
        best: Optional[Tuple[str, int, float]] = None
        for label, record in self._point_records.items():
            state: VariantState = record["state"]  # type: ignore[assignment]
            if not state.visible:
                continue
            x_values: np.ndarray = record["x"]  # type: ignore[assignment]
            y_values: np.ndarray = record["y"]  # type: ignore[assignment]
            if x_values.size == 0:
                continue
            data_xy = np.column_stack((x_values, y_values))
            display = self.ax.transData.transform(data_xy)
            deltas = display - event_xy
            dist_sq = np.einsum("ij,ij->i", deltas, deltas)
            if dist_sq.size == 0:
                continue
            idx = int(np.argmin(dist_sq))
            distance = float(dist_sq[idx])
            if best is None or distance < best[2]:
                best = (label, idx, distance)
        if best is None or best[2] > self._hover_threshold_sq:
            return None
        label, idx, _ = best
        record = self._point_records[label]
        x_values = record["x"]  # type: ignore[assignment]
        y_values = record["y"]  # type: ignore[assignment]
        labels = record["labels"]  # type: ignore[assignment]
        hemispheres = record["hemispheres"]  # type: ignore[assignment]
        return StereographicPayload(
            variant_label=label,
            x=float(x_values[idx]),
            y=float(y_values[idx]),
            text=str(labels[idx]),
            hemisphere=bool(hemispheres[idx]),
            index=idx,
        )

    def _on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button != 1:
            return
        payload = self._nearest_point(event)
        if payload is None:
            return
        suffix = " (S)" if not payload.hemisphere else ""
        text = f"{payload.variant_label}: {payload.text}{suffix}"
        self._annotation_manager.annotate(payload.x, payload.y, text)

    def _handle_toggle(self, label: str) -> None:
        visible = self.variant_manager.toggle(label)
        artists = self._scatter_by_label.get(label, {})
        for artist in artists.values():
            if artist is not None:
                artist.set_visible(visible)
        if self._legend is not None:
            for handle in self._legend.legendHandles:  # type: ignore[attr-defined]
                if handle.get_label() == label:
                    handle.set_alpha(1.0 if visible else 0.2)
        self.figure.canvas.draw_idle()

    def save(self, path: Path | str, *, dpi: Optional[int] = None) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"bbox_inches": "tight"}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        self.figure.savefig(output, **save_kwargs)
        return output

    def as_image_bytes(self, format: str = "png") -> bytes:
        buffer = BytesIO()
        dpi = int(self.settings.figure.get("dpi", 120))
        self.figure.savefig(buffer, format=format, dpi=dpi, bbox_inches="tight")
        return buffer.getvalue()

    def show(self) -> None:
        plt.show()


__all__ = ["StereographicFigure"]
