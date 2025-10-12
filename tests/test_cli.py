from __future__ import annotations

import pytest
from typer.testing import CliRunner

from pycrystallography.cli.main import app


pytestmark = pytest.mark.filterwarnings("ignore:Issues encountered while parsing CIF")


def test_cli_tem_dry_run(sample_config, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "composite",
            "tem",
            "--relation",
            "burgers-zr",
            "--config",
            str(sample_config),
            "--out",
            str(tmp_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "Dry run" in result.stdout


def test_cli_powder_dry_run(sample_config, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "powder",
            "xrd",
            "--phase",
            "alpha-zr",
            "--config",
            str(sample_config),
            "--out",
            str(tmp_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "Dry run" in result.stdout
    assert "Powder pattern computed for alpha-zr" in result.stdout


def test_cli_or_map(sample_config):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "or",
            "map",
            "--relation",
            "burgers-zr",
            "--config",
            str(sample_config),
        ],
    )
    assert result.exit_code == 0
    assert "Variant burgers-zr-v01" in result.stdout
    assert "Direction [1 -1 1] -> [1 -1 0]" in result.stdout


def test_cli_plot_stereographic(sample_config):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "plot",
            "stereographic",
            "--relation",
            "burgers-zr",
            "--config",
            str(sample_config),
            "--no-show",
        ],
    )
    assert result.exit_code == 0
    assert "Prepared stereographic pattern" in result.stdout
    assert "Display suppressed" in result.stdout
