from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pycrystallography.adapters.pymatgen_adapter import DiffractionData, StructureLoader


REGISTRY = Path(__file__).resolve().parents[1] / "data" / "registry.yaml"


def test_tem_camera_length_precision(monkeypatch):
    loader = StructureLoader.from_yaml(REGISTRY)
    structure = loader.get("zr_alpha")
    captured: dict[str, object] = {}

    class DummyCalculator:
        def __init__(self, *, symprec, voltage, beam_direction, camera_length):
            captured["camera_length"] = camera_length
            captured["beam_direction"] = tuple(beam_direction)

        def get_pattern(self, structure):  # pragma: no cover - deterministic stub
            return {
                "Interplanar Spacing": [1.0],
                "Intensity (norm)": [1.0],
                "(hkl)": ["(0, 0, 0)"],
            }

    monkeypatch.setattr(
        "pycrystallography.adapters.pymatgen_adapter.TEMCalculator", DummyCalculator
    )

    data = DiffractionData(voltage=200.0, camera_length=157.25)
    g_values, intensities, hkls = data.tem_pattern(structure, (0, 0, 1))

    assert captured["camera_length"] == pytest.approx(157.25)
    assert captured["beam_direction"] == (0, 0, 1)
    assert np.allclose(g_values, [1.0])
    assert np.allclose(intensities, [1.0])
    assert hkls == [(0, 0, 0)]
