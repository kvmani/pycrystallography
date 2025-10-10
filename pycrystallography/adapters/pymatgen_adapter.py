"""Adapters around :mod:`pymatgen` for deterministic structure IO and diffraction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml  # type: ignore[import-untyped]
from pymatgen.analysis.diffraction.tem import TEMCalculator
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure


@dataclass(slots=True)
class StructureLoader:
    """Load :class:`~pymatgen.core.Structure` instances from a registry."""

    registry: Mapping[str, Path]

    @classmethod
    def from_yaml(cls, path: Path) -> "StructureLoader":
        registry = yaml.safe_load(path.read_text())
        mapping: Dict[str, Path] = {
            key: (path.parent / value).resolve() for key, value in registry.items()
        }
        return cls(registry=mapping)

    def get(self, key_or_path: str | Path) -> Structure:
        if isinstance(key_or_path, (str, Path)):
            key = str(key_or_path)
            if key in self.registry:
                return Structure.from_file(self.registry[key])
            candidate = Path(key)
            if candidate.exists():
                return Structure.from_file(candidate)
        raise KeyError(f"Unknown structure identifier: {key_or_path}")


@dataclass(slots=True)
class DiffractionData:
    """Wrapper around :class:`~pymatgen.analysis.diffraction.tem.TEMCalculator`."""

    voltage: float = 200.0
    camera_length: float = 160.0
    wavelength: float = 1.5406
    symprec: Optional[float] = None

    def tem_pattern(
        self,
        structure: Structure,
        zone_axis: Sequence[int],
        *,
        intensity_threshold: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, Sequence[Tuple[int, int, int]]]:
        """Return scattering vector magnitudes ``g`` and intensities."""

        axis = tuple(int(v) for v in zone_axis)
        if len(axis) != 3:
            raise ValueError("zone_axis must have exactly three indices")
        calculator = TEMCalculator(
            symprec=self.symprec,
            voltage=self.voltage,
            beam_direction=axis,  # type: ignore[arg-type]
            camera_length=int(self.camera_length),
        )
        pattern = calculator.get_pattern(structure)
        d_spacings = np.asarray(pattern["Interplanar Spacing"], dtype=float)
        g_values = 1.0 / d_spacings
        intensities = np.asarray(pattern["Intensity (norm)"], dtype=float)
        hkls = [tuple(int(part) for part in str(hkl).strip("() ").split(",")) for hkl in pattern["(hkl)"]]
        mask = intensities >= intensity_threshold
        filtered_hkls = [hkls[i] for i in np.nonzero(mask)[0]]
        return g_values[mask], intensities[mask], filtered_hkls

    def powder_pattern(
        self,
        structure: Structure,
        *,
        two_theta_range: Optional[Tuple[float, float]] = None,
        min_intensity: float = 0.0,
        debye_waller_factors: Optional[Mapping[int, float]] = None,
        wavelength: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Sequence[Tuple[int, ...]]]:
        """Return ``2θ`` angles, intensities and ``d``-spacings for powder XRD.

        Parameters
        ----------
        structure
            The crystal structure to analyse.
        two_theta_range
            Optional ``(min, max)`` tuple limiting the peak search range.
        min_intensity
            Discard reflections weaker than this relative intensity.
        debye_waller_factors
            Optional Debye-Waller factors keyed by atomic number.
        wavelength
            Override the calculator wavelength (Å). Defaults to the instance
            attribute.
        """

        calculator = XRDCalculator(
            wavelength=wavelength if wavelength is not None else self.wavelength,
            symprec=self.symprec,
            debye_waller_factors=debye_waller_factors,
        )
        pattern = calculator.get_pattern(structure, two_theta_range=two_theta_range)
        two_theta = np.asarray(pattern.x, dtype=float)
        intensities = np.asarray(pattern.y, dtype=float)
        d_spacings = np.asarray(pattern.d_hkls, dtype=float)
        hkls: list[Tuple[int, ...]] = []
        for hkl_group in pattern.hkls:
            if not hkl_group:
                hkls.append(tuple())
                continue
            primary = hkl_group[0]["hkl"]
            hkls.append(tuple(int(value) for value in primary))
        if min_intensity > 0:
            mask = intensities >= float(min_intensity)
            indices = np.nonzero(mask)[0]
            two_theta = two_theta[mask]
            intensities = intensities[mask]
            d_spacings = d_spacings[mask]
            hkls = [hkls[index] for index in indices]
        return two_theta, intensities, d_spacings, tuple(hkls)
