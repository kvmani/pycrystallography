"""Interactive crystallographic diffraction figure."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.widgets import CheckButtons

from ..core.models import CompositePattern
from ..core.variant_manager import VariantManager, VariantState
from .annotation import AnnotationConfig, AnnotationManager
from .settings import PlotSettings


@dataclass(slots=True)
class HoverPayload:
    variant_label: str
    g_value: float
    intensity: float
    d_spacing: float
    hkl: Optional[Tuple[int, ...]]
    index: int


class Tooltip:
    def __init__(self, ax: Axes, settings: Dict[str, object]) -> None:
        self._ax = ax
        self._annotation = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": settings.get("background", "#f8f9fa"),
                "ec": settings.get("border", "#1f2933"),
                "alpha": 0.95,
            },
            fontsize=settings.get("font_size", 9),
            visible=False,
        )
        self._fields: Sequence[str] = tuple(settings.get("fields", ("variant", "g", "intensity")))

    def hide(self) -> None:
        if self._annotation.get_visible():
            self._annotation.set_visible(False)
            self._ax.figure.canvas.draw_idle()

    def show(self, payload: HoverPayload, event) -> None:  # type: ignore[no-untyped-def]
        lines: List[str] = []
        for field in self._fields:
            if field == "variant":
                lines.append(f"Variant: {payload.variant_label}")
            elif field == "g":
                lines.append(f"g = {payload.g_value:.4f} Å⁻¹")
            elif field == "intensity":
                lines.append(f"I = {payload.intensity:.3f}")
            elif field == "d_spacing":
                lines.append(f"d = {payload.d_spacing:.4f} Å")
            elif field == "hkl" and payload.hkl is not None:
                formatted = "(" + " ".join(str(v) for v in payload.hkl) + ")"
                lines.append(f"hkl {formatted}")
        if not lines:
            return
        self._annotation.set_text("\n".join(lines))
        self._annotation.xy = (payload.g_value, payload.intensity)
        self._annotation.set_visible(True)
        self._ax.figure.canvas.draw_idle()


class CrystallographicFigure:
    def __init__(
        self,
        pattern: CompositePattern,
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
        self._tooltip = Tooltip(self.ax, self.settings.tooltip)
        annotation_cfg = AnnotationConfig(
            max_labels=int(self.settings.annotations.get("max_labels", 40)),
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
        self._scatter_by_label: Dict[str, PathCollection] = {}
        self._point_records: Dict[str, Dict[str, object]] = {}
        self._check_buttons: Optional[CheckButtons] = None
        self._legend = None
        self._prepare_points()
        self._init_widgets()
        self._connect_events()

    def _create_figure(self) -> Tuple[Figure, Axes]:
        figure_kwargs = dict(self.settings.figure)
        figsize = figure_kwargs.pop("figsize", (8.0, 6.0))
        dpi = figure_kwargs.pop("dpi", 120)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        if "facecolor" in figure_kwargs:
            fig.patch.set_facecolor(figure_kwargs["facecolor"])
        ax.set_xlabel(r"$g$ (1/Å)")
        ax.set_ylabel("Relative intensity")
        ax.set_title(self.pattern.identifier)
        ax.grid(True, linestyle="--", alpha=0.3)
        return fig, ax

    def _prepare_points(self) -> None:
        scatter_kwargs = dict(self.settings.scatter)
        for state in self.variant_manager:
            indices = np.nonzero(self.pattern.variant_labels == state.variant.label)[0]
            g_values = self.pattern.q_values[indices]
            intensities = self.pattern.intensities[indices]
            d_spacings = self.pattern.d_spacings[indices]
            hkls = None
            if self.pattern.hkls is not None:
                hkls = [self.pattern.hkls[idx] for idx in indices]
            artist = self.ax.scatter(
                g_values,
                intensities,
                marker=state.style.marker,
                c=state.style.color,
                label=state.variant.label,
                visible=state.visible,
                s=scatter_kwargs.get("size", 40),
                alpha=scatter_kwargs.get("alpha", 0.9),
                linewidths=scatter_kwargs.get("linewidth", 0.6),
                edgecolors=scatter_kwargs.get("edgecolor", "#222222"),
            )
            self._scatter_by_label[state.variant.label] = artist
            record = {
                "state": state,
                "g": g_values,
                "intensity": intensities,
                "d": d_spacings,
                "hkls": hkls,
                "artist": artist,
                "indices": indices,
            }
            self._point_records[state.variant.label] = record
        handles = list(self._scatter_by_label.values())
        labels = [state.variant.label for state in self.variant_manager]
        if handles:
            self._legend = self.ax.legend(handles, labels, title="Variant", loc="best")

    def _init_widgets(self) -> None:
        widget_cfg = self.settings.widgets
        anchor = widget_cfg.get("anchor", (0.82, 0.6))
        width = float(widget_cfg.get("width", 0.18))
        height = float(widget_cfg.get("height", 0.25))
        labels = [state.variant.label for state in self.variant_manager]
        states = [state.visible for state in self.variant_manager]
        checkbox_ax = self.figure.add_axes([anchor[0], anchor[1], width, height])
        self._check_buttons = CheckButtons(checkbox_ax, labels, states)
        self._check_buttons.on_clicked(self._handle_toggle)
        checkbox_ax.set_title("Variants")

    def _connect_events(self) -> None:
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)

    def _handle_toggle(self, label: str) -> None:
        visible = self.variant_manager.toggle(label)
        artist = self._scatter_by_label[label]
        artist.set_visible(visible)
        if self._legend is not None:
            for handle in self._legend.legend_handles:
                if handle.get_label() == label:
                    handle.set_alpha(1.0 if visible else 0.2)
        self.figure.canvas.draw_idle()

    def _nearest_point(self, event) -> Optional[HoverPayload]:  # type: ignore[no-untyped-def]
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        event_xy = np.array([event.x, event.y])
        best: Optional[Tuple[str, int, float]] = None
        for label, record in self._point_records.items():
            state: VariantState = record["state"]  # type: ignore[assignment]
            if not state.visible:
                continue
            g_values: np.ndarray = record["g"]  # type: ignore[assignment]
            intensities: np.ndarray = record["intensity"]  # type: ignore[assignment]
            if not len(g_values):
                continue
            data_xy = np.column_stack((g_values, intensities))
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
        label, idx, _distance = best
        record = self._point_records[label]
        state = record["state"]  # type: ignore[assignment]
        g_values: np.ndarray = record["g"]  # type: ignore[assignment]
        intensities: np.ndarray = record["intensity"]  # type: ignore[assignment]
        d_values: np.ndarray = record["d"]  # type: ignore[assignment]
        hkls: Optional[List[Tuple[int, ...]]] = record.get("hkls")  # type: ignore[assignment]
        hkl_value = hkls[idx] if hkls and idx < len(hkls) else None
        return HoverPayload(
            variant_label=state.variant.label,
            g_value=float(g_values[idx]),
            intensity=float(intensities[idx]),
            d_spacing=float(d_values[idx]),
            hkl=hkl_value,
            index=idx,
        )

    def _on_hover(self, event) -> None:  # type: ignore[no-untyped-def]
        payload = self._nearest_point(event)
        if payload is None:
            self._tooltip.hide()
            return
        self._tooltip.show(payload, event)

    def _on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button != 1:
            return
        payload = self._nearest_point(event)
        if payload is None:
            return
        text = f"{payload.variant_label}: g={payload.g_value:.3f}"
        if payload.hkl is not None:
            text = "(" + " ".join(str(part) for part in payload.hkl) + ")"
        self._annotation_manager.annotate(payload.g_value, payload.intensity, text)

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


__all__ = ["CrystallographicFigure"]
