from __future__ import annotations

from pycrystallography.config import load_config


def test_env_override(sample_config, monkeypatch):
    monkeypatch.setenv("PYCG_TEM__VOLTAGE", "210")
    cfg = load_config(sample_config)
    assert cfg.tem.voltage == 210.0
    assert cfg.find_phase("beta-zr").structure == "zr_beta"
