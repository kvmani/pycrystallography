"""Variant visibility and styling management."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Sequence

from .models import Variant


@dataclass(frozen=True, slots=True)
class VariantStyle:
    marker: str
    color: str


@dataclass(slots=True)
class VariantState:
    variant: Variant
    visible: bool
    style: VariantStyle


class MarkerPalette:
    """Assign deterministic markers and colours to variants."""

    def __init__(
        self,
        *,
        shapes: Sequence[str],
        colors: Sequence[str],
        fallback_shape: str = "o",
        fallback_color: str = "#444444",
    ) -> None:
        if not shapes or not colors:
            raise ValueError("MarkerPalette requires at least one shape and colour")
        self._shapes = list(shapes)
        self._colors = list(colors)
        self._fallback = VariantStyle(marker=fallback_shape, color=fallback_color)
        self._assignments: Dict[str, VariantStyle] = {}
        self._pairs = [(shape, colour) for shape in self._shapes for colour in self._colors]
        self._pair_index = 0

    def assign(self, key: str, *, marker: Optional[str] = None, color: Optional[str] = None) -> VariantStyle:
        if key in self._assignments:
            return self._assignments[key]
        if marker is None or color is None:
            if self._pairs:
                default_shape, default_color = self._pairs[self._pair_index % len(self._pairs)]
                self._pair_index += 1
            else:
                default_shape, default_color = self._fallback.marker, self._fallback.color
            shape = marker if marker is not None else default_shape
            colour = color if color is not None else default_color
        else:
            shape, colour = marker, color
            self._pair_index += 1
        style = VariantStyle(marker=shape, color=colour)
        self._assignments[key] = style
        return style

    def __len__(self) -> int:
        return len(self._assignments)


class VariantManager:
    """Stateful controller tracking variant visibility and styling."""

    def __init__(
        self,
        variants: Sequence[Variant],
        *,
        default_visibility: Optional[Mapping[str, bool]] = None,
        palette: Optional[MarkerPalette] = None,
        overrides: Optional[Mapping[str, Mapping[str, str | bool]]] = None,
    ) -> None:
        if not variants:
            raise ValueError("VariantManager requires at least one variant")
        if palette is None:
            palette = MarkerPalette(shapes=["o"], colors=["#1f77b4"])
        self._palette = palette
        self._states: List[VariantState] = []
        visibility_map = dict(default_visibility or {})
        overrides_map = overrides or {}
        for variant in variants:
            override = overrides_map.get(variant.label, {})
            marker_value = override.get("marker")
            color_value = override.get("color")
            marker = marker_value if isinstance(marker_value, str) else None
            color = color_value if isinstance(color_value, str) else None
            visible = visibility_map.get(variant.label, True)
            visible_override = override.get("visible")
            if isinstance(visible_override, bool):
                visible = visible_override
            style = self._palette.assign(variant.label, marker=marker, color=color)
            self._states.append(VariantState(variant=variant, visible=visible, style=style))

    def __iter__(self) -> Iterator[VariantState]:
        return iter(self._states)

    def visible_variants(self) -> List[VariantState]:
        return [state for state in self._states if state.visible]

    def set_visibility(self, label: str, visible: bool) -> None:
        for state in self._states:
            if state.variant.label == label:
                state.visible = visible
                return
        raise KeyError(f"Unknown variant label '{label}'")

    def toggle(self, label: str) -> bool:
        for state in self._states:
            if state.variant.label == label:
                state.visible = not state.visible
                return state.visible
        raise KeyError(f"Unknown variant label '{label}'")

    def all_labels(self) -> List[str]:
        return [state.variant.label for state in self._states]

    def visibility_map(self) -> Dict[str, bool]:
        return {state.variant.label: state.visible for state in self._states}

    def update_overrides(self, overrides: Mapping[str, Mapping[str, str | bool]]) -> None:
        for label, values in overrides.items():
            for state in self._states:
                if state.variant.label != label:
                    continue
                marker_value = values.get("marker")
                color_value = values.get("color")
                visible_value = values.get("visible")
                assign_marker = marker_value if isinstance(marker_value, str) else None
                assign_color = color_value if isinstance(color_value, str) else None
                if assign_marker is not None or assign_color is not None:
                    style = self._palette.assign(
                        label,
                        marker=assign_marker,
                        color=assign_color,
                    )
                    state.style = style
                if isinstance(visible_value, bool):
                    state.visible = visible_value
                break

    def reset(self, *, visible: bool = True) -> None:
        for state in self._states:
            state.visible = visible


__all__ = [
    "MarkerPalette",
    "VariantManager",
    "VariantState",
    "VariantStyle",
]
