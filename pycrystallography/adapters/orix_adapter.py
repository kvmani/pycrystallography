"""Helpers for working with :mod:`orix` orientations and symmetry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from orix.quaternion.orientation import Orientation
from orix.quaternion.symmetry import Symmetry, get_point_group

from ..core.models import OrientationRelation, Variant, ensure_variants_unique
from ..core.indexing import ensure_column_vector


@dataclass(slots=True)
class OrientationFactory:
    """Create :class:`~orix.quaternion.orientation.Orientation` instances."""

    singular_value_tol: float = 1e-8

    def from_direction_pairs(
        self,
        parent_vectors: Sequence[np.ndarray],
        child_vectors: Sequence[np.ndarray],
    ) -> Orientation:
        if len(parent_vectors) < 2 or len(child_vectors) < 2:
            raise ValueError("At least two direction pairs are required")
        if len(parent_vectors) != len(child_vectors):
            raise ValueError("Parent and child vector counts must match")

        parent = np.array([ensure_column_vector(v) for v in parent_vectors], dtype=float)
        child = np.array([ensure_column_vector(v) for v in child_vectors], dtype=float)

        parent_norm = np.linalg.norm(parent, axis=1)
        child_norm = np.linalg.norm(child, axis=1)
        if np.any(parent_norm == 0) or np.any(child_norm == 0):
            raise ValueError("Direction vectors must be non-zero")

        parent_unit = parent / parent_norm[:, None]
        child_unit = child / child_norm[:, None]

        covariance = child_unit.T @ parent_unit
        u, singular_values, vt = np.linalg.svd(covariance)

        if np.count_nonzero(singular_values > self.singular_value_tol) < 2:
            raise ValueError("Direction pairs must span at least two unique directions")

        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = vt.T @ u.T
        return Orientation.from_matrix(rotation)


@dataclass(slots=True)
class VariantGenerator:
    """Generate symmetry-equivalent variants for a given OR."""

    parent_symmetry: Symmetry
    child_symmetry: Symmetry

    @classmethod
    def from_space_groups(cls, parent: int, child: int, *, proper: bool = True) -> "VariantGenerator":
        return cls(
            parent_symmetry=get_point_group(parent, proper=proper),
            child_symmetry=get_point_group(child, proper=proper),
        )

    def generate(self, orientation_relation: OrientationRelation) -> Sequence[Variant]:
        base_orientation = orientation_relation.orientation
        sym_orientations = self.parent_symmetry * base_orientation
        canonical = Orientation(sym_orientations.data.copy())
        canonical.symmetry = self.child_symmetry
        reduced = canonical.map_into_symmetry_reduced_zone()
        _, indices = reduced.unique(return_index=True)
        ordered_indices = np.sort(indices)
        variants = tuple(
            Variant(
                index=i + 1,
                orientation=sym_orientations[idx],
                orientation_relation=orientation_relation,
            )
            for i, idx in enumerate(ordered_indices)
        )
        return ensure_variants_unique(variants)
