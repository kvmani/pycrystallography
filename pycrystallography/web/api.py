"""FastAPI application exposing diffraction services."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from .config import load_ui_config
from .models import (
    GenerateRequest,
    GenerateResponse,
    PowderPatternResponse,
    StructureResponse,
    TemPatternResponse,
    TemRequest,
    UiConfig,
    XrdRequest,
)
from .services import (
    compute_generate,
    compute_tem_pattern,
    compute_powder_pattern,
    structure_from_cif_text,
)


def _decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to decode CIF file")


def create_app(*, allowed_origins: Sequence[str] | None = None, ui_config_path: Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="PyCrystallography Diffraction API",
        version=__version__,
        description=(
            "Backend services for the interactive diffraction workbench. Provides CIF parsing, "
            "powder diffraction, and TEM diffraction capabilities powered by pymatgen."
        ),
    )

    origins = list(allowed_origins or ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ui_config = load_ui_config(ui_config_path)
    app.state.ui_config = ui_config

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ui/config", response_model=UiConfig)
    async def fetch_ui_config(config: UiConfig = Depends(lambda: app.state.ui_config)) -> UiConfig:
        return config

    @app.post("/structures/from-cif", response_model=StructureResponse)
    async def load_structure_from_cif(file: UploadFile = File(...)) -> StructureResponse:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        try:
            text = _decode_bytes(payload)
            response = structure_from_cif_text(text, name=file.filename)
        except Exception as exc:  # pragma: no cover - errors are mapped to HTTP
            raise HTTPException(status_code=400, detail=f"Failed to parse CIF: {exc}") from exc
        return response

    @app.post("/diffraction/xrd", response_model=PowderPatternResponse)
    async def compute_xrd(payload: XrdRequest) -> PowderPatternResponse:
        try:
            return compute_powder_pattern(payload)
        except Exception as exc:  # pragma: no cover - validated in tests
            raise HTTPException(status_code=400, detail=f"XRD calculation failed: {exc}") from exc

    @app.post("/diffraction/tem", response_model=TemPatternResponse)
    async def compute_tem(payload: TemRequest) -> TemPatternResponse:
        try:
            return compute_tem_pattern(payload)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=400, detail=f"TEM calculation failed: {exc}") from exc

    @app.post("/diffraction/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest) -> GenerateResponse:
        try:
            return compute_generate(payload)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=400, detail=f"Diffraction generation failed: {exc}") from exc

    return app


__all__ = ["create_app"]
