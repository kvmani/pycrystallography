from __future__ import annotations

from types import SimpleNamespace

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
