"""Registry for diffraction calculators."""
from __future__ import annotations

import inspect
from importlib import metadata

from .calculators.base import CalculatorRegistry, DiffractionCalculator
from .calculators.composite import CompositeTEMCalculator
from ..adapters.pymatgen_adapter import DiffractionData


calculator_registry = CalculatorRegistry(calculators={})


def _register_builtins() -> None:
    calculator_registry.register(CompositeTEMCalculator(diffraction=DiffractionData()))


def _load_entry_points() -> None:
    for entry_point in metadata.entry_points(group="pycrystallography.calculators"):
        factory = entry_point.load()
        if isinstance(factory, DiffractionCalculator):
            calculator = factory
        elif inspect.isclass(factory) and issubclass(factory, DiffractionCalculator):
            try:
                calculator = factory()
            except TypeError:
                calculator = factory(diffraction=DiffractionData())
        else:
            calculator = factory()
            if not isinstance(calculator, DiffractionCalculator):
                raise TypeError(
                    f"Entry point '{entry_point.name}' did not return a DiffractionCalculator"
                )
        if calculator.slug in calculator_registry.calculators:
            continue
        calculator_registry.register(calculator)


_register_builtins()
_load_entry_points()


def get_calculator(slug: str) -> DiffractionCalculator:
    return calculator_registry.get(slug)
