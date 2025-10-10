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
