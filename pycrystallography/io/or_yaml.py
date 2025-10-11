"""Orientation-relation YAML ingestion and validation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from ..adapters.orix_adapter import OrientationFactory, VariantGenerator
from ..core.indexing import (
    direction_to_cartesian,
    plane_normal_to_cartesian,
)
from ..core.models import OrientationRelation, Phase, Variant
from ..io.cif import StructureCache


class VariantDefaults(BaseModel):
    visible: bool = True
    marker: Optional[str] = None
    color: Optional[str] = None


class VariantSymmetry(BaseModel):
    parent: int = Field(..., ge=1)
    child: int = Field(..., ge=1)


class VariantOverrides(BaseModel):
    symmetry: Optional[VariantSymmetry] = None
    defaults: VariantDefaults = Field(default_factory=VariantDefaults)


class PlotOverrides(BaseModel):
    marker_cycle: Optional[Tuple[str, ...]] = None
    color_cycle: Optional[Tuple[str, ...]] = None
    visibility: Dict[str, bool] = Field(default_factory=dict)


def _normalise_indices(values: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in values)


class OrientationRelationSpec(BaseModel):
    name: str
    hkl_parent: Tuple[int, ...]
    uvw_parent: Tuple[int, ...]
    hkl_product: Tuple[int, ...]
    uvw_product: Tuple[int, ...]
    Euler_parent: Optional[Tuple[float, float, float]] = None
    Euler_product: Optional[Tuple[float, float, float]] = None
    description: Optional[str] = None
    variants: VariantOverrides = Field(default_factory=VariantOverrides)
    plot: Optional[PlotOverrides] = None

    @field_validator("hkl_parent", "uvw_parent", "hkl_product", "uvw_product")
    @classmethod
    def _validate_indices(cls, values: Sequence[int]) -> Tuple[int, ...]:
        if len(values) not in {3, 4}:
            raise ValueError("Indices must have three or four components")
        return _normalise_indices(values)

    @field_validator("Euler_parent", "Euler_product")
    @classmethod
    def _validate_euler(cls, value: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float]]:
        if value is None:
            return None
        if len(value) != 3:
            raise ValueError("Euler angles must contain three values")
        return tuple(float(v) for v in value)

    def build_orientation(
        self,
        parent_phase: Phase,
        child_phase: Phase,
        *,
        orientation_factory: Optional[OrientationFactory] = None,
    ) -> OrientationRelation:
        factory = orientation_factory or OrientationFactory()
        parent_vectors = [
            direction_to_cartesian(parent_phase.structure, self.uvw_parent),
            plane_normal_to_cartesian(parent_phase.structure, self.hkl_parent),
        ]
        child_vectors = [
            direction_to_cartesian(child_phase.structure, self.uvw_product),
            plane_normal_to_cartesian(child_phase.structure, self.hkl_product),
        ]
        orientation = factory.from_direction_pairs(parent_vectors, child_vectors)
        return OrientationRelation(
            name=self.name,
            parent_phase=parent_phase,
            child_phase=child_phase,
            orientation=orientation,
            description=self.description,
        )

    def generate_variants(
        self,
        relation: OrientationRelation,
        *,
        default_symmetry: Tuple[int, int],
    ) -> Tuple[Variant, ...]:
        parent_sg, child_sg = default_symmetry
        if self.variants.symmetry is not None:
            parent_sg = self.variants.symmetry.parent
            child_sg = self.variants.symmetry.child
        generator = VariantGenerator.from_space_groups(parent_sg, child_sg)
        variants = generator.generate(relation)
        return tuple(variants)


class LatticeParameters(BaseModel):
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None


class OrientationDocument(BaseModel):
    phase: str
    cif: str
    space_group: Optional[int] = None
    lattice_parameters: Optional[LatticeParameters] = None
    orientation_relations: Tuple[OrientationRelationSpec, ...]
    plot: Optional[PlotOverrides] = None

    @model_validator(mode="after")
    def _unique_names(self) -> "OrientationDocument":
        names = [spec.name for spec in self.orientation_relations]
        if len(names) != len(set(names)):
            raise ValueError("Orientation relation names must be unique within the document")
        return self


@dataclass(slots=True)
class OrientationLibraryEntry:
    document: OrientationDocument
    structure: StructureSummary

    def build_phase(self) -> Phase:
        metadata = {
            "space_group": self.structure.space_group,
            **self.structure.lattice_parameters,
        }
        warnings = []
        if self.document.lattice_parameters:
            warnings.extend(
                self.structure.validate_metadata(
                    self.document.lattice_parameters.model_dump(exclude_none=True)
                )
            )
        if self.document.space_group is not None:
            warnings.extend(
                self.structure.validate_metadata({"space_group": self.document.space_group})
            )
        meta = {"warnings": warnings, "source": str(self.structure.path)}
        metadata.update(meta)
        return Phase(
            name=self.document.phase,
            structure=self.structure.structure,
            metadata=metadata,
        )


@dataclass(slots=True)
class OrientationLibrary:
    entries: Dict[str, OrientationLibraryEntry]

    def get_phase(self, name: str) -> Phase:
        entry = self.entries[name]
        return entry.build_phase()

    def build_orientation(
        self,
        name: str,
        *,
        parent_phase: Phase,
        child_phase_name: str,
    ) -> Tuple[OrientationRelation, Tuple[Variant, ...], OrientationRelationSpec]:
        entry = self.entries[child_phase_name]
        spec = None
        for candidate in entry.document.orientation_relations:
            if candidate.name == name:
                spec = candidate
                break
        if spec is None:
            raise KeyError(f"Orientation relation '{name}' not found for phase '{child_phase_name}'")
        child_phase = entry.build_phase()
        relation = spec.build_orientation(parent_phase, child_phase)
        variants = spec.generate_variants(
            relation,
            default_symmetry=(
                int(parent_phase.metadata.get("space_group", entry.structure.space_group)),
                entry.structure.space_group,
            ),
        )
        return relation, variants, spec


def load_orientation_document(path: Path) -> OrientationDocument:
    data = yaml.safe_load(path.read_text())
    try:
        return OrientationDocument.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Failed to validate OR YAML at {path}: {exc}") from exc


def load_orientation_library(
    documents: Iterable[Path],
    *,
    cache: Optional[StructureCache] = None,
) -> OrientationLibrary:
    cache = cache or StructureCache()
    entries: Dict[str, OrientationLibraryEntry] = {}
    for document_path in documents:
        doc = load_orientation_document(document_path)
        structure_path = (document_path.parent / doc.cif).resolve()
        structure = cache.get(structure_path)
        entries[doc.phase] = OrientationLibraryEntry(document=doc, structure=structure)
    return OrientationLibrary(entries=entries)


__all__ = [
    "OrientationDocument",
    "OrientationLibrary",
    "OrientationLibraryEntry",
    "OrientationRelationSpec",
    "PlotOverrides",
    "VariantDefaults",
    "VariantOverrides",
    "load_orientation_document",
    "load_orientation_library",
]
