"""Entry points for running the diffraction web backend."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
import uvicorn

from .api import create_app

cli = typer.Typer(help="Run the PyCrystallography diffraction web service")


@cli.command()
def run(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
    reload: bool = typer.Option(False, "--reload/--no-reload", help="Enable autoreload"),
    ui_config: Optional[Path] = typer.Option(None, "--ui-config", exists=True, help="Override UI config"),
    allowed_origin: Optional[List[str]] = typer.Option(
        None,
        "--allowed-origin",
        help="CORS origins",
        show_default=False,
    ),
) -> None:
    """Launch the FastAPI diffraction backend using :mod:`uvicorn`."""

    origins = list(allowed_origin) if allowed_origin else None
    app = create_app(allowed_origins=origins, ui_config_path=ui_config)
    uvicorn.run(app, host=host, port=port, reload=reload)


def main() -> None:
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
