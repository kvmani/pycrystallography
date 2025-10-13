from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pycrystallography.plotting.crystallographic_figure import CrystallographicFigure


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
    artist = figure._scatter_by_label[label]
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
    assert figure._scatter_by_label[label].get_visible() is False
    assert figure._rotation_textbox is not None
    figure._rotation_textbox.set_val("45")
    figure._handle_rotation_submit("45")
    assert manager.visibility_map()[label] is False
    assert figure._scatter_by_label[label].get_visible() is False
