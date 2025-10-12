from __future__ import annotations

import numpy as np


def test_point_records_preserve_hemisphere_flags(stereographic_bundle):
    variants, pattern, figure = stereographic_bundle
    record = figure._point_records[variants[0].label]
    assert record["hemispheres"].tolist() == [True, False]
    coords = pattern.cartesian_coordinates()
    assert np.all(np.linalg.norm(coords, axis=1) <= 1.05)


def test_lower_hemisphere_markers_are_outlined(stereographic_bundle):
    variants, _pattern, figure = stereographic_bundle
    artists = figure._scatter_by_label[variants[0].label]
    lower_artist = artists["lower"]
    assert lower_artist is not None
    assert lower_artist.get_facecolors().size == 0
    upper_artist = artists["upper"]
    assert upper_artist is not None
    assert upper_artist.get_facecolors().size > 0
