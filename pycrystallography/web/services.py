"""Core services that back the web API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from pymatgen.core import Structure

from ..adapters.pymatgen_adapter import DiffractionData
from .models import (
    GenerateRequest,
    GenerateResponse,
    PowderPatternPoint,
    PowderPatternResponse,
    StructureModel,
    StructureResponse,
    StructureSummary,
    TemPatternResponse,
    TemReflection,
    TemRequest,
    XrdRequest,
)


@dataclass
class DiffractionEngines:
    """Factory for diffraction calculators used by the web service."""

    tem: DiffractionData
    xrd: DiffractionData

    @classmethod
    def create(
        cls,
        *,
        wavelength: float = 1.5406,
        voltage: float = 200.0,
        camera_length: float = 160.0,
        symprec: float | None = None,
    ) -> "DiffractionEngines":
        data = DiffractionData(
            wavelength=wavelength,
            voltage=voltage,
            camera_length=camera_length,
            symprec=symprec,
        )
        return cls(tem=data, xrd=data)


def structure_from_model(model: StructureModel) -> Structure:
    return model.to_structure()


def structure_from_cif_text(content: str, *, name: str | None = None) -> StructureResponse:
    structure = Structure.from_str(content, fmt="cif")
    warnings = list(iter_structure_warnings(structure))
    model = StructureModel.from_structure(structure, name=name)
    summary = StructureSummary.from_structure(structure, warnings=warnings)
    return StructureResponse(structure=model, summary=summary)


def iter_structure_warnings(structure: Structure) -> Iterable[str]:
    if not structure.is_ordered:
        yield "Structure contains disordered sites; occupancies were collapsed to the dominant species."
    for index, site in enumerate(structure.sites, start=1):
        total = float(sum(site.species.values()))
        if not np.isclose(total, 1.0):
            yield f"Site {index} has occupancy sum {total:.3f}."


def compute_powder_pattern(request: XrdRequest) -> PowderPatternResponse:
    structure = request.structure.to_structure()
    engines = DiffractionData(wavelength=request.settings.wavelength)
    two_theta, intensities, d_spacings, hkls = engines.powder_pattern(
        structure,
        two_theta_range=request.settings.to_range(),
        min_intensity=request.settings.min_intensity,
        wavelength=request.settings.wavelength,
    )
    peaks: List[PowderPatternPoint] = []
    for value, intensity, spacing, hkl in zip(two_theta, intensities, d_spacings, hkls, strict=True):
        peaks.append(
            PowderPatternPoint(
                two_theta=float(value),
                intensity=float(intensity),
                d_spacing=float(spacing),
                hkl=tuple(int(v) for v in hkl),
            )
        )
    return PowderPatternResponse(peaks=peaks)


def compute_tem_pattern(request: TemRequest) -> TemPatternResponse:
    structure = request.structure.to_structure()
    engines = DiffractionData(
        voltage=request.settings.voltage,
        camera_length=request.settings.camera_length,
        symprec=None,
    )
    g_values, intensities, hkls, positions = engines.tem_pattern(
        structure,
        zone_axis=request.settings.zone_axis,
        intensity_threshold=request.settings.intensity_threshold,
    )
    reflections = [
        TemReflection(
            g=float(g),
            intensity=float(intensity),
            hkl=tuple(int(v) for v in hkl),
            position=(float(position[0]), float(position[1])),
        )
        for g, intensity, hkl, position in zip(g_values, intensities, hkls, positions, strict=True)
    ]
    return TemPatternResponse(reflections=reflections)


def compute_generate(request: GenerateRequest) -> GenerateResponse:
    structure = request.structure.to_structure()
    warnings = list(iter_structure_warnings(structure))
    summary = StructureSummary.from_structure(structure, warnings=warnings)
    xrd_response = compute_powder_pattern(XrdRequest(structure=request.structure, settings=request.xrd))
    tem_response = compute_tem_pattern(TemRequest(structure=request.structure, settings=request.tem))
    return GenerateResponse(
        structure=request.structure,
        summary=summary,
        xrd=xrd_response,
        tem=tem_response,
    )


__all__ = [
    "DiffractionEngines",
    "compute_generate",
    "compute_powder_pattern",
    "compute_tem_pattern",
    "iter_structure_warnings",
    "structure_from_cif_text",
    "structure_from_model",
]
