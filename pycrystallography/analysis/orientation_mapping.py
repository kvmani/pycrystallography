"""Helpers for mapping parent features onto child variants."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..config.models import IndexSpec
from ..core.indexing import (
    direction_three_to_four,
    indices_to_cartesian,
    plane_three_to_four,
    fractional_to_integer_indices,
)
from ..core.models import OrientationRelation, Variant


def _orientation_matrix(orientation) -> np.ndarray:
    matrix = orientation.to_matrix()
    if matrix.ndim == 3:
        matrix = matrix[0]
    return matrix


def _should_emit_four_index(spec: IndexSpec, structure) -> bool:
    if len(spec.indices) == 4:
        return True
    lattice = structure.lattice
    return bool(getattr(lattice, "is_hexagonal", lambda: False)())


@dataclass(frozen=True)
class FeatureMapping:
    """Description of how a single parent feature maps to the child lattice."""

    spec: IndexSpec
    child_indices: tuple[int, int, int]
    child_indices_four: tuple[int, int, int, int] | None
    child_cartesian: np.ndarray


@dataclass(frozen=True)
class VariantFeatureMapping:
    """Mapping results for a single variant."""

    variant: Variant
    mappings: tuple[FeatureMapping, ...]


def map_parent_features_to_child_variants(
    relation: OrientationRelation,
    variants: Sequence[Variant],
    features: Iterable[IndexSpec],
) -> tuple[VariantFeatureMapping, ...]:
    """Project parent features into the child lattices for each variant."""

    feature_specs = list(features)
    parent_structure = relation.parent_phase.structure
    child_structure = relation.child_phase.structure
    parent_vectors = [
        indices_to_cartesian(parent_structure, kind=spec.kind, indices=spec.indices)
        for spec in feature_specs
    ]
    mappings: list[VariantFeatureMapping] = []
    for variant in variants:
        rotation = _orientation_matrix(variant.orientation)
        variant_mappings: list[FeatureMapping] = []
        for spec, parent_vector in zip(feature_specs, parent_vectors, strict=True):
            child_vector = rotation.T @ parent_vector
            norm = np.linalg.norm(child_vector)
            if norm == 0:
                raise ValueError("Parent feature produced a zero-length child vector")
            unit = child_vector / norm
            if spec.kind == "direction":
                fractional = child_structure.lattice.get_fractional_coords(unit)
                indices = fractional_to_integer_indices(fractional.tolist())
                four = (
                    direction_three_to_four(indices)
                    if _should_emit_four_index(spec, child_structure)
                    else None
                )
            else:
                reciprocal = child_structure.lattice.reciprocal_lattice
                fractional = reciprocal.get_fractional_coords(unit)
                indices = fractional_to_integer_indices(fractional.tolist())
                four = (
                    plane_three_to_four(indices)
                    if _should_emit_four_index(spec, child_structure)
                    else None
                )
            variant_mappings.append(
                FeatureMapping(
                    spec=spec,
                    child_indices=indices,
                    child_indices_four=four,
                    child_cartesian=unit,
                )
            )
        mappings.append(
            VariantFeatureMapping(variant=variant, mappings=tuple(variant_mappings))
        )
    return tuple(mappings)


__all__ = [
    "FeatureMapping",
    "VariantFeatureMapping",
    "map_parent_features_to_child_variants",
]
