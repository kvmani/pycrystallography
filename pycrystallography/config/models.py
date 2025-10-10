"""Pydantic models describing pycrystallography configuration."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from pydantic import BaseModel, Field, model_validator


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
    parent_directions: List[Sequence[float]]
    child_directions: List[Sequence[float]]

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
