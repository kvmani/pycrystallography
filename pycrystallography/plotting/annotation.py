"""Annotation layout helpers for diffraction plots."""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import List, Tuple

from matplotlib.axes import Axes
from matplotlib.text import Annotation


@dataclass(slots=True)
class AnnotationConfig:
    max_labels: int
    min_distance: float
    offset_radius: float
    polar_steps: int
    font_size: float
    box_facecolor: str
    box_edgecolor: str
    box_alpha: float


class AnnotationManager:
    def __init__(self, ax: Axes, config: AnnotationConfig) -> None:
        self._ax = ax
        self._config = config
        self._annotations: List[Annotation] = []

    def clear(self) -> None:
        while self._annotations:
            annotation = self._annotations.pop()
            annotation.remove()

    def _candidate_offsets(self) -> List[Tuple[float, float]]:
        radius = self._config.offset_radius
        steps = max(4, self._config.polar_steps)
        return [
            (radius * cos(2 * pi * i / steps), radius * sin(2 * pi * i / steps))
            for i in range(steps)
        ]

    def _overlaps(self, bbox, other_bbox) -> bool:  # type: ignore[no-untyped-def]
        return not (
            bbox.x1 < other_bbox.x0
            or bbox.x0 > other_bbox.x1
            or bbox.y1 < other_bbox.y0
            or bbox.y0 > other_bbox.y1
        )

    def _find_position(self, x: float, y: float) -> Tuple[float, float]:
        renderer = self._ax.figure.canvas.get_renderer()
        existing = [annotation.get_window_extent(renderer=renderer) for annotation in self._annotations]
        base = self._ax.transData.transform((x, y))
        for dx, dy in self._candidate_offsets():
            candidate = (base[0] + dx, base[1] + dy)
            temp = self._ax.annotate(
                "",
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset pixels",
                visible=False,
            )
            bbox = temp.get_window_extent(renderer=renderer)
            temp.remove()
            if all(not self._overlaps(bbox, other) for other in existing):
                inv = self._ax.transData.inverted()
                return inv.transform(candidate)
        return self._ax.transData.inverted().transform((base[0] + 4, base[1] + 4))

    def annotate(self, x: float, y: float, text: str) -> Annotation:
        if len(self._annotations) >= self._config.max_labels:
            oldest = self._annotations.pop(0)
            oldest.remove()
        position = self._find_position(x, y)
        annotation = self._ax.annotate(
            text,
            xy=(x, y),
            xytext=position,
            textcoords="data",
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": self._config.box_facecolor,
                "ec": self._config.box_edgecolor,
                "alpha": self._config.box_alpha,
            },
            fontsize=self._config.font_size,
            ha="center",
        )
        self._annotations.append(annotation)
        self._ax.figure.canvas.draw_idle()
        return annotation

    def active_annotations(self) -> List[Annotation]:
        return list(self._annotations)


__all__ = ["AnnotationConfig", "AnnotationManager"]
