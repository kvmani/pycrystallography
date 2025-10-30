"""Utilities for loading plotting settings from YAML files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


@dataclass(frozen=True, slots=True)
class PlotSettings:
    figure: Dict[str, Any]
    scatter: Dict[str, Any]
    markers: Dict[str, Any]
    marker_toggles: Dict[str, Dict[str, Any]]
    marker_variant_overrides: Dict[str, Dict[str, Any]]
    annotations: Dict[str, Any]
    tooltip: Dict[str, Any]
    widgets: Dict[str, Any]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PlotSettings":
        def _section(name: str, default: Mapping[str, Any]) -> Dict[str, Any]:
            raw = mapping.get(name, {}) if isinstance(mapping.get(name), Mapping) else {}
            data = dict(default)
            data.update(raw)  # type: ignore[arg-type]
            return data

        figure = _section("figure", {"dpi": 120, "figsize": (8.0, 6.0)})
        scatter = _section(
            "scatter",
            {"size": 40, "alpha": 0.9, "linewidth": 0.6, "edgecolor": "#222222"},
        )
        marker_defaults: Dict[str, Any] = {
            "shapes": ("o", "s", "^", "v"),
            "colors": ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"),
            "fallback_shape": "o",
            "fallback_color": "#444444",
            "toggles": {
                "primary": {
                    "enabled": False,
                    "marker": "o",
                    "color": "#1f77b4",
                    "flag": "is_primary",
                },
                "absent": {
                    "enabled": False,
                    "marker": "X",
                    "color": "#d62728",
                    "flag": "is_absent",
                },
                "kikuchi": {
                    "enabled": False,
                    "marker": "+",
                    "color": "#17becf",
                    "flag": "is_kikuchi",
                },
            },
            "variants": {},
        }
        raw_markers = mapping.get("markers", {})
        markers = dict(marker_defaults)
        if isinstance(raw_markers, Mapping):
            for key, value in raw_markers.items():
                if key in {"shapes", "colors", "fallback_shape", "fallback_color"}:
                    markers[key] = value
                elif key == "toggles" and isinstance(value, Mapping):
                    toggles = {
                        name: dict(config)
                        for name, config in marker_defaults["toggles"].items()
                    }
                    for name, config in value.items():
                        if not isinstance(config, Mapping):
                            continue
                        base = toggles.get(
                            name,
                            {
                                "enabled": False,
                                "marker": markers.get("fallback_shape", "o"),
                                "color": markers.get("fallback_color", "#444444"),
                                "flag": name,
                            },
                        )
                        merged = dict(base)
                        merged.update(config)
                        if "flag" not in merged:
                            merged["flag"] = name
                        toggles[str(name)] = merged
                    markers["toggles"] = toggles
                elif key == "variants" and isinstance(value, Mapping):
                    overrides: Dict[str, Dict[str, Any]] = {}
                    for variant_label, override in value.items():
                        if isinstance(override, Mapping):
                            overrides[str(variant_label)] = dict(override)
                        else:
                            overrides[str(variant_label)] = {}
                    markers["variants"] = overrides
                else:
                    markers[key] = value
        toggles_cfg = {
            name: dict(config)
            for name, config in markers.get("toggles", {}).items()
        }
        for name, config in list(toggles_cfg.items()):
            if "flag" not in config:
                config["flag"] = name
            toggles_cfg[name] = config
        variant_overrides = {
            str(label): dict(config)
            for label, config in markers.get("variants", {}).items()
        }
        annotations = _section(
            "annotations",
            {
                "max_labels": 40,
                "min_distance": 10.0,
                "offset_radius": 10.0,
                "polar_steps": 24,
                "font_size": 9,
                "box_facecolor": "#fffff8",
                "box_edgecolor": "#666666",
                "box_alpha": 0.9,
            },
        )
        tooltip = _section(
            "tooltip",
            {
                "background": "#f8f9fa",
                "border": "#1f2933",
                "font_size": 9,
                "fields": ("variant", "g", "intensity"),
            },
        )
        widgets = _section(
            "widgets",
            {"checkbox_columns": 2, "width": 0.2, "height": 0.25, "anchor": (0.82, 0.6)},
        )
        return cls(
            figure=figure,
            scatter=scatter,
            markers=markers,
            marker_toggles=toggles_cfg,
            marker_variant_overrides=variant_overrides,
            annotations=annotations,
            tooltip=tooltip,
            widgets=widgets,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PlotSettings":
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, Mapping):
            raise TypeError("Plot settings YAML must define a mapping at the top level")
        return cls.from_mapping(raw)


__all__ = ["PlotSettings"]
