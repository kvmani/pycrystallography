from __future__ import annotations

from pycrystallography.core.variant_manager import MarkerPalette, VariantManager


def test_variant_manager_visibility(orientation_bundle):
    _cfg, _relation, variants, _spec, manager, _pattern, _figure = orientation_bundle
    # All variants visible by default
    assert all(state.visible for state in manager)
    first_label = variants[0].label
    manager.set_visibility(first_label, False)
    assert not manager.visibility_map()[first_label]
    manager.toggle(first_label)
    assert manager.visibility_map()[first_label]


def test_marker_palette_assigns_unique_markers(orientation_bundle):
    _cfg, _relation, variants, _spec, _manager, _pattern, _figure = orientation_bundle
    palette = MarkerPalette(shapes=["o", "s"], colors=["#000000", "#ff0000"])
    other_manager = VariantManager(variants, palette=palette)
    styles = [state.style for state in other_manager]
    unique_styles = {style for style in styles}
    max_unique = len({(shape, colour) for shape in ["o", "s"] for colour in ["#000000", "#ff0000"]})
    assert len(unique_styles) == min(len(variants), max_unique)
