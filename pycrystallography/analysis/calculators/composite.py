"""Composite diffraction calculators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from pymatgen.core.operations import SymmOp

from ...adapters.pymatgen_adapter import DiffractionData
from ...core.models import CompositePattern, Variant
from .base import DiffractionCalculator


@dataclass(slots=True)
class CompositeTEMCalculator(DiffractionCalculator):
    """Compute composite Selected Area Electron Diffraction (SAED) patterns."""

    diffraction: DiffractionData
    slug: str = "tem-composite"

    def compute(
        self,
        *,
        variants: Sequence[Variant],
        metadata: Mapping[str, object] | None = None,
    ) -> CompositePattern:
        if not variants:
            raise ValueError("At least one variant is required")
        metadata = dict(metadata or {})
        if "zone_axis" not in metadata:
            raise ValueError("'zone_axis' must be provided in metadata")
        zone_axis_raw = metadata["zone_axis"]
        if not isinstance(zone_axis_raw, Sequence):
            raise TypeError("zone_axis metadata must be a sequence of integers")
        zone_axis = tuple(int(v) for v in zone_axis_raw)

        def _to_float(value: object, default: float) -> float:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return default
            return default

        voltage = _to_float(metadata.get("voltage"), self.diffraction.voltage)
        camera_length = _to_float(metadata.get("camera_length"), self.diffraction.camera_length)
        intensity_threshold = _to_float(metadata.get("intensity_threshold"), 1e-6)
        data = DiffractionData(
            voltage=voltage,
            camera_length=camera_length,
            symprec=self.diffraction.symprec,
        )
        q_values: list[np.ndarray] = []
        intensities: list[np.ndarray] = []
        labels: list[str] = []
        for variant in variants:
            rotated = self._rotate_structure(
                variant.orientation_relation.child_phase.structure, variant.orientation
            )
            g_values, i_values, _hkls = data.tem_pattern(
                rotated, zone_axis, intensity_threshold=intensity_threshold
            )
            q_values.append(g_values)
            intensities.append(i_values)
            labels.extend([variant.label] * len(g_values))
        if q_values:
            q_concat = np.concatenate(q_values)
            intensity_concat = np.concatenate(intensities)
            labels_array = np.array(labels, dtype="U32")
            order = np.argsort(q_concat)
            q_concat = q_concat[order]
            intensity_concat = intensity_concat[order]
            labels_array = labels_array[order]
        else:
            q_concat = np.array([])
            intensity_concat = np.array([])
            labels_array = np.array([], dtype="U32")
        return CompositePattern(
            identifier=str(metadata.get("identifier", "composite")),
            variants=tuple(variants),
            q_values=q_concat,
            intensities=intensity_concat,
            variant_labels=labels_array,
            metadata=metadata,
        )

    @staticmethod
    def _rotate_structure(structure, orientation):
        matrix = orientation.to_matrix()
        if matrix.ndim == 3:
            matrix = matrix[0]
        op = SymmOp.from_rotation_and_translation(matrix, [0, 0, 0])
        rotated = structure.copy()
        rotated.apply_operation(op)
        return rotated
