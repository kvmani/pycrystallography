"""Pydantic models for the diffraction web service."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from pymatgen.core import Lattice, Structure


class LatticeParameters(BaseModel):
    """Unit cell parameters (Å, degrees)."""

    model_config = ConfigDict(extra="forbid")

    a: float = Field(..., gt=0.0)
    b: float = Field(..., gt=0.0)
    c: float = Field(..., gt=0.0)
    alpha: float = Field(..., gt=0.0)
    beta: float = Field(..., gt=0.0)
    gamma: float = Field(..., gt=0.0)

    def to_lattice(self) -> Lattice:
        return Lattice.from_parameters(
            a=float(self.a),
            b=float(self.b),
            c=float(self.c),
            alpha=float(self.alpha),
            beta=float(self.beta),
            gamma=float(self.gamma),
        )


class AtomSite(BaseModel):
    """Fractional coordinates for an atom site."""

    model_config = ConfigDict(extra="forbid")

    element: str = Field(..., min_length=1)
    x: float
    y: float
    z: float
    occupancy: float = Field(1.0, gt=0.0)
    label: Optional[str] = Field(None, description="Site label from CIF if available")

    def fractional_coordinates(self) -> np.ndarray:
        return np.array([float(self.x), float(self.y), float(self.z)], dtype=float)


class StructureModel(BaseModel):
    """Serializable representation of a :class:`pymatgen.core.Structure`."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    lattice: LatticeParameters
    space_group: Optional[str] = None
    atom_sites: List[AtomSite]

    @model_validator(mode="after")
    def _ensure_sites(self) -> "StructureModel":
        if not self.atom_sites:
            msg = "At least one atom site is required to build a structure"
            raise ValueError(msg)
        return self

    def to_structure(self) -> Structure:
        lattice = self.lattice.to_lattice()
        species: List[str | dict[str, float]] = []
        labels: List[str] = []
        for index, site in enumerate(self.atom_sites, start=1):
            if abs(site.occupancy - 1.0) > 1e-6:
                species.append({site.element: float(site.occupancy)})
            else:
                species.append(site.element)
            labels.append(site.label or f"{site.element}{index}")
        frac_coords = [site.fractional_coordinates() for site in self.atom_sites]
        structure = Structure(lattice, species, frac_coords, site_properties={"label": labels})
        return structure

    @classmethod
    def from_structure(cls, structure: Structure, *, name: Optional[str] = None) -> "StructureModel":
        lattice = structure.lattice
        params = LatticeParameters(
            a=float(lattice.a),
            b=float(lattice.b),
            c=float(lattice.c),
            alpha=float(lattice.alpha),
            beta=float(lattice.beta),
            gamma=float(lattice.gamma),
        )
        atom_sites = []
        for site in structure.sites:
            site_label = None
            if "label" in site.properties:
                site_label = str(site.properties["label"])
            composition = site.species.as_dict()
            if len(composition) > 1:
                # collapse into the most populous species for display purposes
                element = max(composition, key=composition.get)
                occupancy = float(composition[element])
            else:
                element = next(iter(composition))
                occupancy = float(next(iter(composition.values())))
            atom_sites.append(
                AtomSite(
                    element=str(element),
                    x=float(site.frac_coords[0]),
                    y=float(site.frac_coords[1]),
                    z=float(site.frac_coords[2]),
                    occupancy=occupancy,
                    label=site_label if site_label else None,
                )
            )
        sg_symbol: Optional[str] = None
        try:
            sg_symbol = structure.get_space_group_info()[0]
        except Exception:
            sg_symbol = None
        return cls(
            name=name,
            lattice=params,
            atom_sites=atom_sites,
            space_group=sg_symbol,
        )


class StructureSummary(BaseModel):
    """Additional metadata about a structure for UI consumption."""

    model_config = ConfigDict(extra="forbid")

    formula: str
    density: float
    volume: float
    lattice_vectors: List[List[float]]
    space_group: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    @classmethod
    def from_structure(cls, structure: Structure, *, warnings: Optional[Sequence[str]] = None) -> "StructureSummary":
        vectors = structure.lattice.matrix.astype(float).tolist()
        space_group: Optional[str] = None
        if getattr(structure, "is_3d_periodic", True):
            try:
                space_group = structure.get_space_group_info()[0]
            except Exception:
                space_group = None
        return cls(
            formula=structure.composition.reduced_formula,
            density=float(structure.density),
            volume=float(structure.volume),
            lattice_vectors=vectors,
            space_group=space_group,
            warnings=list(warnings) if warnings else [],
        )


class StructureResponse(BaseModel):
    """Payload returned after parsing a CIF file."""

    model_config = ConfigDict(extra="forbid")

    structure: StructureModel
    summary: StructureSummary


class XrdSettings(BaseModel):
    """Input options for powder XRD calculations."""

    model_config = ConfigDict(extra="forbid")

    wavelength: float = Field(1.5406, gt=0.0)
    two_theta_min: float = Field(5.0, ge=0.0)
    two_theta_max: float = Field(90.0, gt=0.0)
    min_intensity: float = Field(0.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_range(self) -> "XrdSettings":
        if self.two_theta_max <= self.two_theta_min:
            msg = "two_theta_max must be greater than two_theta_min"
            raise ValueError(msg)
        return self

    def to_range(self) -> Tuple[float, float]:
        return float(self.two_theta_min), float(self.two_theta_max)


class TemSettings(BaseModel):
    """Input options for TEM diffraction calculations."""

    model_config = ConfigDict(extra="forbid")

    zone_axis: Tuple[int, int, int] = Field(default=(0, 0, 1))
    voltage: float = Field(200.0, gt=0.0)
    camera_length: float = Field(160.0, gt=0.0)
    intensity_threshold: float = Field(1e-5, ge=0.0)


class PowderPatternPoint(BaseModel):
    two_theta: float
    intensity: float
    d_spacing: float
    hkl: Tuple[int, ...]


class PowderPatternResponse(BaseModel):
    """Powder diffraction pattern serialised for the frontend."""

    model_config = ConfigDict(extra="forbid")

    peaks: List[PowderPatternPoint]


class TemReflection(BaseModel):
    g: float
    intensity: float
    hkl: Tuple[int, int, int]
    position: Tuple[float, float]


class TemPatternResponse(BaseModel):
    """TEM diffraction pattern response."""

    model_config = ConfigDict(extra="forbid")

    reflections: List[TemReflection]


class GenerateRequest(BaseModel):
    """Combined generation request for the full UI."""

    model_config = ConfigDict(extra="forbid")

    structure: StructureModel
    xrd: XrdSettings
    tem: TemSettings


class GenerateResponse(BaseModel):
    """Combined payload containing all diffraction results."""

    model_config = ConfigDict(extra="forbid")

    structure: StructureModel
    summary: StructureSummary
    xrd: PowderPatternResponse
    tem: TemPatternResponse


class XrdRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structure: StructureModel
    settings: XrdSettings


class TemRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structure: StructureModel
    settings: TemSettings


class ElementColor(BaseModel):
    element: str
    color: str


class UiConfig(BaseModel):
    """UI configuration served to the client."""

    model_config = ConfigDict(extra="forbid")

    background_color: str = "#101418"
    atom_scale: float = Field(0.8, gt=0.0)
    bond_thickness: float = Field(0.06, gt=0.0)
    element_colors: List[ElementColor] = Field(default_factory=list)
    show_bonds: bool = True
    default_supercell: Tuple[int, int, int] = (1, 1, 1)


class UiConfigModel(RootModel[UiConfig]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

