"""Pydantic models describing pycrystallography configuration."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from pydantic import BaseModel, Field, model_validator

from ..core.indexing import validate_miller_bravais


class IndexSpec(BaseModel):
    """Specification of a crystallographic direction or plane."""

    kind: str = Field(default="direction")
    indices: Tuple[float, ...]
    label: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Mapping[str, Any]:
        if isinstance(value, IndexSpec):
            return value.model_dump()
        if isinstance(value, Mapping):
            if "direction" in value:
                return {
                    "kind": "direction",
                    "indices": tuple(value["direction"]),
                    "label": value.get("label"),
                }
            if "plane" in value:
                return {
                    "kind": "plane",
                    "indices": tuple(value["plane"]),
                    "label": value.get("label"),
                }
            if "kind" in value and "indices" in value:
                return {
                    "kind": value["kind"],
                    "indices": tuple(value["indices"]),
                    "label": value.get("label"),
                }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return {"kind": "direction", "indices": tuple(value)}
        raise TypeError(f"Cannot construct IndexSpec from {value!r}")

    @model_validator(mode="after")
    def _validate(self) -> "IndexSpec":
        if self.kind not in {"direction", "plane"}:
            raise ValueError(f"Invalid index kind '{self.kind}'")
        if len(self.indices) not in {3, 4}:
            raise ValueError("Indices must have three or four components")
        if len(self.indices) == 4:
            validate_miller_bravais(self.indices)
        return self


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO", description="Python logging level")


class PhaseConfig(BaseModel):
    name: str
    structure: str
    space_group: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OrientationRelationConfig(BaseModel):
    name: str
    parent_phase: str
    child_phase: str
    parent_directions: List[IndexSpec]
    child_directions: List[IndexSpec]

    @model_validator(mode="after")
    def _check_lengths(self) -> "OrientationRelationConfig":
        if len(self.parent_directions) < 2 or len(self.child_directions) < 2:
            raise ValueError("At least two direction pairs are required for an OR")
        if len(self.parent_directions) != len(self.child_directions):
            raise ValueError("Parent and child direction counts must match")
        return self


class TEMOptions(BaseModel):
    calculator: str = Field(default="tem-composite")
    voltage: float = Field(default=200.0, description="Accelerating voltage in kV")
    camera_length: float = Field(default=160.0, description="Camera length in mm")
    zone_axis: Tuple[int, int, int] = Field(default=(0, 0, 1))
    intensity_threshold: float = Field(default=1e-6)


class AppConfig(BaseModel):
    phases: List[PhaseConfig]
    orientation_relations: List[OrientationRelationConfig]
    tem: TEMOptions = Field(default_factory=TEMOptions)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def find_phase(self, name: str) -> PhaseConfig:
        for phase in self.phases:
            if phase.name == name:
                return phase
        raise KeyError(f"Unknown phase '{name}' in configuration")

    def find_orientation(self, name: str) -> OrientationRelationConfig:
        for relation in self.orientation_relations:
            if relation.name == name:
                return relation
        raise KeyError(f"Unknown orientation relation '{name}'")

    def model_post_init(self, __context: Any) -> None:
        phase_names = {phase.name for phase in self.phases}
        for relation in self.orientation_relations:
            if relation.parent_phase not in phase_names:
                raise ValueError(f"Parent phase '{relation.parent_phase}' is not defined")
            if relation.child_phase not in phase_names:
                raise ValueError(f"Child phase '{relation.child_phase}' is not defined")
