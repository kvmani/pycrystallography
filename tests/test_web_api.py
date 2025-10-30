from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from pymatgen.core import Structure

from pycrystallography.web import create_app
from pycrystallography.web.models import StructureModel

DATA_DIR = Path("data/structureData")
FE_CIF = DATA_DIR / "Fe.cif"
ZR_CIF = DATA_DIR / "Zr-Alpha.cif"


def build_client() -> TestClient:
    return TestClient(create_app())


def load_structure_model(path: Path) -> StructureModel:
    structure = Structure.from_file(path)
    return StructureModel.from_structure(structure, name=path.stem)


def test_cif_upload_returns_structure() -> None:
    client = build_client()
    with FE_CIF.open("rb") as handle:
        response = client.post(
            "/structures/from-cif",
            files={"file": (FE_CIF.name, handle, "application/octet-stream")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["structure"]["atom_sites"], "Expected atom sites in response"
    assert payload["summary"]["formula"] == "Fe"


def test_xrd_endpoint_returns_peaks() -> None:
    client = build_client()
    model = load_structure_model(FE_CIF)
    response = client.post(
        "/diffraction/xrd",
        json={
            "structure": model.model_dump(mode="json"),
            "settings": {
                "wavelength": 1.5406,
                "two_theta_min": 20.0,
                "two_theta_max": 80.0,
                "min_intensity": 0.0,
            },
        },
    )
    assert response.status_code == 200
    peaks = response.json()["peaks"]
    assert len(peaks) > 0
    assert all("two_theta" in peak for peak in peaks)


def test_tem_endpoint_returns_reflections() -> None:
    client = build_client()
    model = load_structure_model(FE_CIF)
    response = client.post(
        "/diffraction/tem",
        json={
            "structure": model.model_dump(mode="json"),
            "settings": {
                "zone_axis": [0, 0, 1],
                "voltage": 200.0,
                "camera_length": 160.0,
                "intensity_threshold": 1e-3,
            },
        },
    )
    assert response.status_code == 200
    reflections = response.json()["reflections"]
    assert len(reflections) > 0
    assert all("g" in reflection for reflection in reflections)
    assert all("position" in reflection for reflection in reflections)


def test_generate_combines_all_outputs() -> None:
    client = build_client()
    model = load_structure_model(ZR_CIF)
    response = client.post(
        "/diffraction/generate",
        json={
            "structure": model.model_dump(mode="json"),
            "xrd": {
                "wavelength": 1.5406,
                "two_theta_min": 10.0,
                "two_theta_max": 90.0,
                "min_intensity": 0.0,
            },
            "tem": {
                "zone_axis": [0, 0, 1],
                "voltage": 200.0,
                "camera_length": 160.0,
                "intensity_threshold": 1e-3,
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["structure"]["atom_sites"]
    assert payload["summary"]["formula"]
    assert payload["xrd"]["peaks"]
    assert payload["tem"]["reflections"]
