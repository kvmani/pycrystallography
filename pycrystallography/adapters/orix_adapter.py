"""Helpers for working with :mod:`orix` orientations and symmetry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from orix.quaternion.orientation import Orientation
from orix.quaternion.symmetry import Symmetry, get_point_group

from ..core.models import OrientationRelation, Variant, ensure_variants_unique


def _orthonormal_basis(vectors: Sequence[np.ndarray]) -> np.ndarray:
    first, second = vectors[:2]
    b1 = first / np.linalg.norm(first)
    v2 = second - np.dot(second, b1) * b1
    b2 = v2 / np.linalg.norm(v2)
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3 / np.linalg.norm(b3)])


@dataclass(slots=True)
class OrientationFactory:
    """Create :class:`~orix.quaternion.orientation.Orientation` instances."""

    def from_direction_pairs(
        self,
        parent_vectors: Sequence[np.ndarray],
        child_vectors: Sequence[np.ndarray],
    ) -> Orientation:
        if len(parent_vectors) < 2 or len(child_vectors) < 2:
            raise ValueError("At least two direction pairs are required")
        parent_basis = _orthonormal_basis(parent_vectors)
        child_basis = _orthonormal_basis(child_vectors)
        rotation = parent_basis @ child_basis.T
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
