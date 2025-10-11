from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from orix.quaternion.orientation import Orientation

from pycrystallography.adapters.orix_adapter import OrientationFactory
from pycrystallography.adapters.pymatgen_adapter import StructureLoader
from pycrystallography.config import IndexSpec, load_config
from pycrystallography.core.indexing import direction_to_cartesian, plane_normal_to_cartesian


REGISTRY = Path(__file__).resolve().parents[1] / "data" / "registry.yaml"


def test_invalid_miller_bravais_indices() -> None:
    with pytest.raises(ValueError):
        IndexSpec(kind="direction", indices=(1, 1, 1, 0))


def test_direction_plane_conversion(sample_config) -> None:
    cfg = load_config(sample_config)
    loader = StructureLoader.from_yaml(REGISTRY)
    structure = loader.get(cfg.find_phase("alpha-zr").structure)

    direction = direction_to_cartesian(structure, (1, -1, 0, 0))
    plane = plane_normal_to_cartesian(structure, (0, 0, 0, 1))

    assert np.linalg.norm(direction) > 0
    assert np.linalg.norm(plane) > 0
    assert np.isclose(np.dot(direction, plane), 0.0, atol=1e-6)


def test_orientation_factory_recovers_rotation() -> None:
    factory = OrientationFactory()
    orientation = Orientation.from_euler((25.0, 40.0, 10.0), degrees=True)
    rotation = orientation.to_matrix()
    if rotation.ndim == 3:
        rotation = rotation[0]

    child_vectors = [
        np.array([1.0, 0.2, 0.1], dtype=float),
        np.array([0.0, 1.0, 0.3], dtype=float),
        np.array([0.2, 0.1, 1.0], dtype=float),
    ]
    parent_vectors = [rotation @ vec for vec in child_vectors]

    result = factory.from_direction_pairs(parent_vectors, child_vectors)
    result_matrix = result.to_matrix()
    if result_matrix.ndim == 3:
        result_matrix = result_matrix[0]

    for expected, child in zip(parent_vectors, child_vectors):
        assert np.allclose(result_matrix @ child, expected, atol=1e-7)


def test_orientation_factory_colinear_error() -> None:
    factory = OrientationFactory()
    parent = [np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]
    child = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 2.0, 0.0])]
    with pytest.raises(ValueError, match="span at least two"):
        factory.from_direction_pairs(parent, child)
