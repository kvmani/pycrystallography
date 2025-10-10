"""Domain models for pycrystallography."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
from orix.quaternion.orientation import Orientation
from pymatgen.core import Structure


@dataclass(frozen=True, slots=True)
class Phase:
    """A crystallographic phase with a known crystal structure."""

    name: str
    structure: Structure
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_metadata(self, **updates: Any) -> "Phase":
        metadata: MutableMapping[str, Any] = dict(self.metadata)
        metadata.update(updates)
        return Phase(name=self.name, structure=self.structure, metadata=metadata)


@dataclass(frozen=True, slots=True)
class OrientationRelation:
    """Describes the orientation relationship (OR) between two phases."""

    name: str
    parent_phase: Phase
    child_phase: Phase
    orientation: Orientation
    description: Optional[str] = None


@dataclass(frozen=True, slots=True)
class Variant:
    """A symmetry-equivalent variant generated from an OR."""

    index: int
    orientation: Orientation
    orientation_relation: OrientationRelation

    @property
    def label(self) -> str:
        return f"{self.orientation_relation.name}-v{self.index:02d}"


@dataclass(slots=True)
class CompositePattern:
    """Container for the result of a composite diffraction calculation."""

    identifier: str
    variants: Sequence[Variant]
    q_values: np.ndarray
    intensities: np.ndarray
    variant_labels: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_table(self) -> np.ndarray:
        """Return a structured array suitable for saving to CSV/NPZ."""

        dtype = [("q", float), ("intensity", float), ("variant", "U32")]
        return np.array(
            list(zip(self.q_values, self.intensities, self.variant_labels)),
            dtype=dtype,
        )

    def copy_with(self, *, metadata: Optional[Mapping[str, Any]] = None) -> "CompositePattern":
        meta = dict(self.metadata)
        if metadata:
            meta.update(metadata)
        return CompositePattern(
            identifier=self.identifier,
            variants=self.variants,
            q_values=np.array(self.q_values, copy=True),
            intensities=np.array(self.intensities, copy=True),
            variant_labels=np.array(self.variant_labels, copy=True),
            metadata=meta,
        )


@dataclass(frozen=True, slots=True)
class PowderPattern:
    """Powder X-ray diffraction pattern for a single phase."""

    phase: Phase
    two_theta: np.ndarray
    intensities: np.ndarray
    d_spacings: np.ndarray
    hkls: Sequence[Sequence[int]]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_table(self) -> np.ndarray:
        """Return a structured array of peak data."""

        dtype = [
            ("two_theta", float),
            ("intensity", float),
            ("d_spacing", float),
            ("hkl", "U32"),
        ]
        hkl_strings = [
            " ".join(str(part) for part in hkl) if hkl else ""
            for hkl in self.hkls
        ]
        return np.array(
            list(zip(self.two_theta, self.intensities, self.d_spacings, hkl_strings)),
            dtype=dtype,
        )


def ensure_variants_unique(variants: Iterable[Variant]) -> Sequence[Variant]:
    """Ensure that variant labels are unique and deterministic."""

    seen: Dict[str, Variant] = {}
    for variant in variants:
        if variant.label in seen:
            raise ValueError(f"Duplicate variant label detected: {variant.label}")
        seen[variant.label] = variant
    return tuple(seen.values())
