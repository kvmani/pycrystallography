"""Configuration loading utilities."""
from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml  # type: ignore[import-untyped]

from .models import AppConfig

ENV_PREFIX = "PYCG_"


def _merge_dicts(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _merge_dicts(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _load_from_env(env: Mapping[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, raw in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX) :].lower().split("__")
        cursor: MutableMapping[str, Any] = result
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})  # type: ignore[assignment]
        cursor[path[-1]] = _parse_env_value(raw)
    return result


def _parse_env_value(raw: str) -> Any:
    for parser in (json.loads, float, str):
        try:
            return parser(raw)
        except Exception:
            continue
    return raw


def _load_file(path: Path) -> Mapping[str, Any]:
    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    if path.suffix == ".json":
        return json.loads(text)
    if path.suffix == ".toml":
        return tomllib.loads(text)
    raise ValueError(f"Unsupported configuration format: {path.suffix}")


def load_config(
    path: Path | None,
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    data: MutableMapping[str, Any] = {}
    if path:
        data = dict(_load_file(path))
    env_data = _load_from_env(env or os.environ)
    data = _merge_dicts(data, env_data)
    if overrides:
        data = _merge_dicts(data, overrides)
    return AppConfig.model_validate(data)


def resolve_config(
    *,
    config_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    return load_config(config_path, env=env, overrides=overrides)
