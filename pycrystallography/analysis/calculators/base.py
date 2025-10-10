"""Base classes for diffraction calculators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

from ...core.models import CompositePattern, Variant


class DiffractionCalculator(ABC):
    """Abstract base class for diffraction calculators."""

    slug: str

    @abstractmethod
    def compute(
        self,
        *,
        variants: Sequence[Variant],
        metadata: Mapping[str, object] | None = None,
    ) -> CompositePattern:
        """Compute a composite diffraction pattern."""

    def describe(self) -> str:
        return self.slug


@dataclass(slots=True)
class CalculatorRegistry:
    """In-memory registry for calculator plug-ins."""

    calculators: MutableMapping[str, DiffractionCalculator]

    def register(self, calculator: DiffractionCalculator) -> None:
        if calculator.slug in self.calculators:
            raise ValueError(f"Calculator '{calculator.slug}' already registered")
        self.calculators[calculator.slug] = calculator

    def get(self, slug: str) -> DiffractionCalculator:
        if slug not in self.calculators:
            raise KeyError(f"Unknown calculator '{slug}'")
        return self.calculators[slug]

    def available(self) -> Sequence[str]:
        return tuple(sorted(self.calculators))
