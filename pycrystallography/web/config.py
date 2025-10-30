"""Configuration helpers for the diffraction web app."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import yaml

from .models import UiConfig

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "diffraction_web_ui.yaml"


def load_ui_config(path: Optional[Path] = None) -> UiConfig:
    """Load UI configuration from ``path`` or the default config file."""

    candidate = path or _DEFAULT_CONFIG
    if candidate.exists():
        data = yaml.safe_load(candidate.read_text())
        if not isinstance(data, Mapping):
            raise ValueError("UI config must be a mapping")
        return UiConfig.model_validate(data)
    return UiConfig()


__all__ = ["load_ui_config"]
