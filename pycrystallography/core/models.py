"""Domain models for pycrystallography."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

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
    hkls: Optional[Sequence[Tuple[int, ...]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_table(self) -> np.ndarray:
        """Return a structured array suitable for saving to CSV/NPZ."""

        dtype = [("g", float), ("intensity", float), ("variant", "U32"), ("d_spacing", float), ("hkl", "U32")]
        rows = []
        d_spacings = self.d_spacings
        for idx, label in enumerate(self.variant_labels):
            hkl = ""
            if self.hkls is not None and idx < len(self.hkls):
                hkl_tuple = self.hkls[idx]
                if hkl_tuple:
                    hkl = "(" + " ".join(str(part) for part in hkl_tuple) + ")"
            rows.append(
                (
                    float(self.q_values[idx]),
                    float(self.intensities[idx]),
                    str(label),
                    float(d_spacings[idx]),
                    hkl,
                )
            )
        return np.array(rows, dtype=dtype)

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
            hkls=tuple(self.hkls) if self.hkls is not None else None,
            metadata=meta,
        )

    @property
    def d_spacings(self) -> np.ndarray:
        with np.errstate(divide="ignore"):
            d_values = np.where(self.q_values != 0, 1.0 / self.q_values, 0.0)
        return d_values


@dataclass(slots=True)
class StereographicPattern:
    """Container describing poles for stereographic projection plots."""

    identifier: str
    variants: Sequence[Variant]
    polar_angles: np.ndarray
    variant_labels: np.ndarray
    hemispheres: np.ndarray
    labels: Sequence[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.polar_angles = np.asarray(self.polar_angles, dtype=float)
        if self.polar_angles.ndim != 2 or self.polar_angles.shape[1] != 2:
            raise ValueError("polar_angles must be an (N, 2) array of (theta, phi)")
        count = int(self.polar_angles.shape[0])
        self.variant_labels = np.asarray(self.variant_labels, dtype=str)
        self.hemispheres = np.asarray(self.hemispheres, dtype=bool)
        if self.variant_labels.shape[0] != count or self.hemispheres.shape[0] != count:
            raise ValueError("variant_labels and hemispheres must match polar_angles length")
        if len(self.labels) != count:
            raise ValueError("labels must match polar_angles length")
        self.labels = tuple(str(label) for label in self.labels)
        self.metadata = dict(self.metadata)

    @property
    def theta(self) -> np.ndarray:
        return self.polar_angles[:, 0]

    @property
    def phi(self) -> np.ndarray:
        return self.polar_angles[:, 1]

    def cartesian_coordinates(self) -> np.ndarray:
        """Return projected X/Y coordinates for each pole."""

        radii = np.tan(self.theta / 2.0)
        x = radii * np.cos(self.phi)
        y = radii * np.sin(self.phi)
        return np.column_stack((x, y))


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
