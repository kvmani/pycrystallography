"""Interactive crystallographic diffraction figure."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.widgets import CheckButtons, TextBox

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
        if self.settings.marker_variant_overrides:
            self.variant_manager.update_overrides(
                self.settings.marker_variant_overrides
            )
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
        self._scatter_by_label: Dict[str, List[PathCollection]] = {}
        self._point_records: Dict[str, Dict[str, object]] = {}
        self._check_buttons: Optional[CheckButtons] = None
        self._rotation_textbox: Optional[TextBox] = None
        self._rotation_angle: float = 0.0
        self._rotation_factory: Optional[Callable[[float], CompositePattern]] = (
            self._resolve_rotation_factory(pattern.metadata)
        )
        self._base_rotation_vectors: Optional[np.ndarray] = self._extract_rotation_vectors(
            pattern.metadata
        )
        if (
            self._base_rotation_vectors is not None
            and self._base_rotation_vectors.shape[0] != len(self.pattern.q_values)
        ):
            raise ValueError(
                "rotation_vectors must have the same number of rows as pattern points"
            )
        self._base_rotation_norms: Optional[np.ndarray] = None
        if self._base_rotation_vectors is not None:
            self._base_rotation_norms = np.linalg.norm(
                self._base_rotation_vectors, axis=1
            )
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
        toggles = self.settings.marker_toggles
        toggle_order = list(toggles.keys())
        handles: List[PathCollection] = []
        legend_labels: List[str] = []
        for state in self.variant_manager:
            indices = np.nonzero(self.pattern.variant_labels == state.variant.label)[0]
            if not len(indices):
                self._scatter_by_label[state.variant.label] = []
                self._point_records[state.variant.label] = {
                    "state": state,
                    "indices": indices,
                    "g": np.array([], dtype=float),
                    "intensity": np.array([], dtype=float),
                    "d": np.array([], dtype=float),
                    "hkls": [] if self.pattern.hkls is not None else None,
                    "subsets": [],
                }
                continue
            g_values = self.pattern.q_values[indices]
            intensities = self.pattern.intensities[indices]
            d_spacings = self.pattern.d_spacings[indices]
            hkls = None
            if self.pattern.hkls is not None:
                hkls = [self.pattern.hkls[idx] for idx in indices]
            assignments = np.full(indices.shape, "__default__", dtype=object)
            assigned = np.zeros(indices.shape, dtype=bool)
            for name in toggle_order:
                config = toggles.get(name, {})
                if not bool(config.get("enabled", False)):
                    continue
                flag_name = str(config.get("flag", name))
                flag_array = self.pattern.get_flag(flag_name)
                if flag_array is None:
                    continue
                slice_mask = np.asarray(flag_array[indices], dtype=bool)
                if slice_mask.shape != assignments.shape:
                    continue
                category_mask = np.logical_and(slice_mask, ~assigned)
                if not np.any(category_mask):
                    continue
                assignments[category_mask] = name
                assigned |= category_mask
            subsets: List[Dict[str, object]] = []
            artists: List[PathCollection] = []
            categories: List[str] = []
            if np.any(assignments == "__default__"):
                categories.append("__default__")
            for name in toggle_order:
                if np.any(assignments == name):
                    categories.append(name)
            for category in categories:
                mask = assignments == category
                subset_indices = indices[mask]
                subset_g = g_values[mask]
                subset_intensity = intensities[mask]
                subset_d = d_spacings[mask]
                subset_hkls = None
                if hkls is not None:
                    subset_hkls = [hkls[i] for i, selected in enumerate(mask) if selected]
                marker = state.style.marker
                color = state.style.color
                if category != "__default__":
                    config = toggles.get(category, {})
                    marker = str(config.get("marker", marker))
                    color = str(config.get("color", color))
                label = state.variant.label if not subsets else None
                artist = self.ax.scatter(
                    subset_g,
                    subset_intensity,
                    marker=marker,
                    c=color,
                    label=label,
                    visible=state.visible,
                    s=scatter_kwargs.get("size", 40),
                    alpha=scatter_kwargs.get("alpha", 0.9),
                    linewidths=scatter_kwargs.get("linewidth", 0.6),
                    edgecolors=scatter_kwargs.get("edgecolor", "#222222"),
                )
                artists.append(artist)
                subsets.append(
                    {
                        "indices": subset_indices,
                        "g": subset_g,
                        "intensity": subset_intensity,
                        "d": subset_d,
                        "hkls": subset_hkls,
                        "category": category,
                        "artist": artist,
                    }
                )
            self._scatter_by_label[state.variant.label] = artists
            record = {
                "state": state,
                "indices": indices,
                "g": g_values,
                "intensity": intensities,
                "d": d_spacings,
                "hkls": hkls,
                "subsets": subsets,
            }
            self._point_records[state.variant.label] = record
            if artists:
                handles.append(artists[0])
                legend_labels.append(state.variant.label)
        if handles:
            self._legend = self.ax.legend(handles, legend_labels, title="Variant", loc="best")

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
        rotation_cfg = widget_cfg.get("rotation", {})
        rot_anchor = rotation_cfg.get(
            "anchor", (anchor[0], min(anchor[1] + height + 0.02, 0.95))
        )
        rot_width = float(rotation_cfg.get("width", width))
        rot_height = float(rotation_cfg.get("height", 0.05))
        rotation_ax = self.figure.add_axes([rot_anchor[0], rot_anchor[1], rot_width, rot_height])
        label = str(rotation_cfg.get("label", "Rotation (°)"))
        initial = str(rotation_cfg.get("initial", f"{self._rotation_angle:.1f}"))
        self._rotation_textbox = TextBox(rotation_ax, label, initial=initial)
        self._rotation_textbox.on_submit(self._handle_rotation_submit)

    def _connect_events(self) -> None:
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)

    def _handle_toggle(self, label: str) -> None:
        visible = self.variant_manager.toggle(label)
        artists = self._scatter_by_label.get(label, [])
        for artist in artists:
            artist.set_visible(visible)
        if self._legend is not None:
            for handle in self._legend.legend_handles:
                if handle.get_label() == label:
                    handle.set_alpha(1.0 if visible else 0.2)
        self.figure.canvas.draw_idle()

    def _handle_rotation_submit(self, text: str) -> None:
        try:
            angle = float(text)
        except (TypeError, ValueError):
            if self._rotation_textbox is not None:
                self._rotation_textbox.eventson = False
                self._rotation_textbox.set_val(f"{self._rotation_angle:.3f}")
                self._rotation_textbox.eventson = True
            return
        self._apply_rotation(angle)

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

    def _apply_rotation(self, angle: float) -> None:
        if np.isclose(angle, self._rotation_angle):
            return
        magnitudes: Optional[np.ndarray] = None
        if self._rotation_factory is not None:
            rotated = self._rotation_factory(angle)
            if not isinstance(rotated, CompositePattern):
                raise TypeError("Rotation factory must return a CompositePattern")
            self.pattern = rotated
            self._base_rotation_vectors = self._extract_rotation_vectors(rotated.metadata)
            if self._base_rotation_vectors is not None:
                self._base_rotation_norms = np.linalg.norm(
                    self._base_rotation_vectors, axis=1
                )
            else:
                self._base_rotation_norms = None
            q_values = np.asarray(rotated.q_values, dtype=float)
            intensities = np.asarray(rotated.intensities, dtype=float)
            hkls = rotated.hkls
            variant_labels = np.asarray(rotated.variant_labels)
        else:
            q_values = self._rotated_q_values(angle)
            if q_values is None:
                if self._rotation_textbox is not None:
                    self._rotation_textbox.eventson = False
                    self._rotation_textbox.set_val(f"{self._rotation_angle:.3f}")
                    self._rotation_textbox.eventson = True
                return
            intensities = np.asarray(self.pattern.intensities, dtype=float)
            hkls = self.pattern.hkls
            variant_labels = np.asarray(self.pattern.variant_labels)
            self.pattern.q_values = q_values
            magnitudes = self._base_rotation_norms
        self._update_point_records(
            q_values,
            intensities,
            hkls,
            variant_labels,
            magnitudes=magnitudes,
        )
        self._rotation_angle = angle
        if self._rotation_textbox is not None:
            self._rotation_textbox.eventson = False
            self._rotation_textbox.set_val(f"{self._rotation_angle:.3f}")
            self._rotation_textbox.eventson = True
        self.ax.relim()
        self.ax.autoscale_view()
        self.figure.canvas.draw_idle()

    def _update_point_records(
        self,
        q_values: np.ndarray,
        intensities: np.ndarray,
        hkls: Optional[Sequence[Tuple[int, ...]]],
        variant_labels: np.ndarray,
        *,
        magnitudes: Optional[np.ndarray] = None,
    ) -> None:
        if magnitudes is not None:
            d_spacings = np.zeros_like(magnitudes, dtype=float)
            mask = magnitudes != 0
            d_spacings[mask] = 1.0 / magnitudes[mask]
        else:
            absolute_q = np.abs(q_values)
            d_spacings = np.zeros_like(absolute_q, dtype=float)
            mask = absolute_q != 0
            d_spacings[mask] = 1.0 / absolute_q[mask]
        hkls_list: Optional[List[Tuple[int, ...]]] = None
        if hkls is not None:
            hkls_list = [tuple(hkl) for hkl in hkls]
        toggles = self.settings.marker_toggles
        toggle_order = list(toggles.keys())
        for label, record in self._point_records.items():
            indices = np.nonzero(variant_labels == label)[0]
            record["indices"] = indices
            g_slice = q_values[indices]
            intensity_slice = intensities[indices]
            d_slice = d_spacings[indices]
            record["g"] = g_slice
            record["intensity"] = intensity_slice
            record["d"] = d_slice
            if hkls_list is not None:
                record["hkls"] = [hkls_list[idx] for idx in indices]
            assignments = np.full(indices.shape, "__default__", dtype=object)
            assigned = np.zeros(indices.shape, dtype=bool)
            for name in toggle_order:
                config = toggles.get(name, {})
                if not bool(config.get("enabled", False)):
                    continue
                flag_name = str(config.get("flag", name))
                flag_array = self.pattern.get_flag(flag_name)
                if flag_array is None:
                    continue
                slice_mask = np.asarray(flag_array[indices], dtype=bool)
                if slice_mask.shape != assignments.shape:
                    continue
                category_mask = np.logical_and(slice_mask, ~assigned)
                if not np.any(category_mask):
                    continue
                assignments[category_mask] = name
                assigned |= category_mask
            subsets = record.get("subsets")
            if not isinstance(subsets, list):
                subsets = []
                record["subsets"] = subsets
            category_order: List[str] = []
            if np.any(assignments == "__default__"):
                category_order.append("__default__")
            for name in toggle_order:
                if np.any(assignments == name):
                    category_order.append(name)
            subset_map: Dict[str, Dict[str, object]] = {
                str(subset.get("category")): subset
                for subset in subsets
                if isinstance(subset, dict) and "category" in subset
            }
            for category in category_order:
                mask = assignments == category
                subset_indices = indices[mask]
                subset_g = g_slice[mask]
                subset_intensity = intensity_slice[mask]
                subset_d = d_slice[mask]
                subset_hkls = None
                if hkls_list is not None:
                    subset_hkls = [hkls_list[idx] for idx in subset_indices]
                subset = subset_map.get(category)
                if subset is None:
                    continue
                subset["indices"] = subset_indices
                subset["g"] = subset_g
                subset["intensity"] = subset_intensity
                subset["d"] = subset_d
                subset["hkls"] = subset_hkls
                artist = subset.get("artist")
                if isinstance(artist, PathCollection):
                    offsets = np.column_stack((subset_g, subset_intensity))
                    artist.set_offsets(offsets)
            for category, subset in subset_map.items():
                if category in category_order:
                    continue
                subset["indices"] = np.array([], dtype=int)
                subset["g"] = np.array([], dtype=float)
                subset["intensity"] = np.array([], dtype=float)
                subset["d"] = np.array([], dtype=float)
                subset["hkls"] = [] if hkls_list is not None else None
                artist = subset.get("artist")
                if isinstance(artist, PathCollection):
                    artist.set_offsets(np.empty((0, 2)))

    def _rotated_q_values(self, angle: float) -> Optional[np.ndarray]:
        if self._base_rotation_vectors is None:
            return None
        vectors = self._base_rotation_vectors
        if vectors.ndim != 2 or vectors.shape[1] < 2:
            return None
        radians = np.deg2rad(angle)
        rotation = np.array(
            [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
        )
        rotated_xy = vectors[:, :2] @ rotation.T
        return rotated_xy[:, 0]

    @staticmethod
    def _resolve_rotation_factory(
        metadata: Mapping[str, object]
    ) -> Optional[Callable[[float], CompositePattern]]:
        factory = metadata.get("rotation_factory") if isinstance(metadata, Mapping) else None
        return factory if callable(factory) else None

    @staticmethod
    def _extract_rotation_vectors(metadata: Mapping[str, object]) -> Optional[np.ndarray]:
        if not isinstance(metadata, Mapping):
            return None
        vectors = metadata.get("rotation_vectors")
        if vectors is None:
            return None
        array = np.asarray(vectors, dtype=float)
        if array.ndim != 2:
            raise ValueError("rotation_vectors must be a 2D array")
        return np.array(array, copy=True)


__all__ = ["CrystallographicFigure"]
