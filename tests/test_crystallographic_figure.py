from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pycrystallography.core.variant_manager import MarkerPalette, VariantManager
from pycrystallography.plotting.crystallographic_figure import CrystallographicFigure
from pycrystallography.plotting.settings import PlotSettings


def _make_event(ax, x, y):
    display = ax.transData.transform((x, y))
    return SimpleNamespace(
        inaxes=ax,
        x=display[0],
        y=display[1],
        xdata=x,
        ydata=y,
        button=1,
    )


def _marker_signature(collection):
    paths = collection.get_paths()
    if not paths:
        return tuple()
    return tuple(paths[0].vertices.flatten())


def test_hover_payload_matches_variant(orientation_bundle):
    _cfg, relation, variants, spec, manager, pattern, figure = orientation_bundle
    event = _make_event(figure.ax, pattern.q_values[0], pattern.intensities[0])
    payload = figure._nearest_point(event)
    assert payload is not None
    assert payload.variant_label == variants[0].label


def test_click_creates_annotation(orientation_bundle):
    _cfg, relation, variants, spec, manager, pattern, figure = orientation_bundle
    event = _make_event(figure.ax, pattern.q_values[1], pattern.intensities[1])
    figure._on_click(event)
    annotations = figure._annotation_manager.active_annotations()
    assert annotations
    assert annotations[0].get_visible()


def test_rotation_widget_updates_scatter_offsets(orientation_bundle):
    _cfg, _relation, variants, _spec, _manager, pattern, figure = orientation_bundle
    label = variants[0].label
    artist = figure._scatter_by_label[label][0]
    original_offsets = np.array(artist.get_offsets())
    base_vectors = np.array(pattern.metadata["rotation_vectors"], copy=True)
    assert figure._rotation_textbox is not None
    figure._rotation_textbox.set_val("90")
    figure._handle_rotation_submit("90")
    rotated_offsets = np.array(artist.get_offsets())
    indices = np.nonzero(pattern.variant_labels == label)[0]
    expected = -base_vectors[indices, 1]
    np.testing.assert_allclose(rotated_offsets[:, 0], expected)
    np.testing.assert_allclose(rotated_offsets[:, 1], original_offsets[:, 1])


def test_rotation_preserves_variant_visibility(orientation_bundle):
    _cfg, _relation, variants, _spec, manager, _pattern, figure = orientation_bundle
    label = variants[0].label
    figure._handle_toggle(label)
    assert manager.visibility_map()[label] is False
    assert all(art.get_visible() is False for art in figure._scatter_by_label[label])
    assert figure._rotation_textbox is not None
    figure._rotation_textbox.set_val("45")
    figure._handle_rotation_submit("45")
    assert manager.visibility_map()[label] is False
    assert all(art.get_visible() is False for art in figure._scatter_by_label[label])


def test_primary_toggle_changes_marker(orientation_bundle):
    _cfg, _relation, variants, _spec, _manager, pattern, _figure = orientation_bundle
    base_settings = PlotSettings.from_mapping(
        {"markers": {"shapes": ["o"], "colors": ["#1f77b4"], "toggles": {"primary": {"enabled": False}}}}
    )
    base_palette = MarkerPalette(
        shapes=base_settings.markers.get("shapes", ("o",)),
        colors=base_settings.markers.get("colors", ("#1f77b4",)),
    )
    base_manager = VariantManager(variants, palette=base_palette)
    base_manager.update_overrides(base_settings.marker_variant_overrides)
    base_figure = CrystallographicFigure(
        pattern.copy_with(),
        base_manager,
        settings=base_settings,
        backend="Agg",
    )
    label = variants[0].label
    base_collections = base_figure._scatter_by_label[label]
    assert len(base_collections) == 1
    base_signature = _marker_signature(base_collections[0])

    primary_settings = PlotSettings.from_mapping(
        {
            "markers": {
                "shapes": ["o"],
                "colors": ["#1f77b4"],
                "toggles": {"primary": {"enabled": True, "marker": "s", "color": "#ff0000"}},
            }
        }
    )
    primary_palette = MarkerPalette(
        shapes=primary_settings.markers.get("shapes", ("o",)),
        colors=primary_settings.markers.get("colors", ("#1f77b4",)),
    )
    primary_manager = VariantManager(variants, palette=primary_palette)
    primary_manager.update_overrides(primary_settings.marker_variant_overrides)
    primary_figure = CrystallographicFigure(
        pattern.copy_with(),
        primary_manager,
        settings=primary_settings,
        backend="Agg",
    )
    subset_records = primary_figure._point_records[label]["subsets"]
    primary_subset = next(
        subset
        for subset in subset_records
        if subset.get("category") == "primary"
    )
    flagged_collection = primary_subset["artist"]
    flagged_signature = _marker_signature(flagged_collection)
    assert flagged_signature and flagged_signature != base_signature


def test_absent_toggle_changes_marker(orientation_bundle):
    _cfg, _relation, variants, _spec, _manager, pattern, _figure = orientation_bundle
    base_settings = PlotSettings.from_mapping(
        {"markers": {"shapes": ["o"], "colors": ["#1f77b4"], "toggles": {"absent": {"enabled": False}}}}
    )
    base_palette = MarkerPalette(
        shapes=base_settings.markers.get("shapes", ("o",)),
        colors=base_settings.markers.get("colors", ("#1f77b4",)),
    )
    base_manager = VariantManager(variants, palette=base_palette)
    base_manager.update_overrides(base_settings.marker_variant_overrides)
    base_figure = CrystallographicFigure(
        pattern.copy_with(),
        base_manager,
        settings=base_settings,
        backend="Agg",
    )
    label = variants[0].label
    absent_settings = PlotSettings.from_mapping(
        {
            "markers": {
                "shapes": ["o"],
                "colors": ["#1f77b4"],
                "toggles": {"absent": {"enabled": True, "marker": "X", "color": "#000000"}},
            }
        }
    )
    absent_palette = MarkerPalette(
        shapes=absent_settings.markers.get("shapes", ("o",)),
        colors=absent_settings.markers.get("colors", ("#1f77b4",)),
    )
    absent_manager = VariantManager(variants, palette=absent_palette)
    absent_manager.update_overrides(absent_settings.marker_variant_overrides)
    absent_figure = CrystallographicFigure(
        pattern.copy_with(),
        absent_manager,
        settings=absent_settings,
        backend="Agg",
    )
    subset_records = absent_figure._point_records[label]["subsets"]
    absent_subset = next(
        subset
        for subset in subset_records
        if subset.get("category") == "absent"
    )
    absent_collection = absent_subset["artist"]
    base_signature = _marker_signature(base_figure._scatter_by_label[label][0])
    absent_signature = _marker_signature(absent_collection)
    assert absent_signature and absent_signature != base_signature
