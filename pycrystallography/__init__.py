"""Public package interface for :mod:`pycrystallography`.

This module exposes the supported public API and wires together
configuration loading utilities, core domain models and the plug-in
registry for diffraction calculators.
"""
from __future__ import annotations

from importlib import metadata

from .analysis.registry import calculator_registry, get_calculator
from .config.loader import load_config, resolve_config
from .config.models import AppConfig
from .core.models import CompositePattern, OrientationRelation, Phase, PowderPattern, Variant

__all__ = [
    "AppConfig",
    "CompositePattern",
    "OrientationRelation",
    "Phase",
    "PowderPattern",
    "Variant",
    "calculator_registry",
    "get_calculator",
    "load_config",
    "resolve_config",
]

try:
    __version__ = metadata.version("pycrystallography")
except metadata.PackageNotFoundError:  # pragma: no cover - local dev
    __version__ = "0.0.0"
