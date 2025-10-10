from __future__ import annotations

from typer.testing import CliRunner

from pycrystallography.cli.main import app


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
