import json
from pathlib import Path
from typing import Iterable

from pymatgen.analysis.diffraction.tem import TEMCalculator
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "structureData"
FIXTURE_DIR = ROOT / "webapp" / "tests" / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

xrd_calculator = XRDCalculator(debye_waller_factors=None)


def structure_to_model(structure: Structure, name: str | None = None) -> dict:
    lattice = structure.lattice
    atom_sites = []
    for index, site in enumerate(structure.sites, start=1):
        specie = site.species_string.split()[0]
        occupancy = float(max(site.species.values())) if site.species else 1.0
        atom_sites.append(
            {
                "element": specie,
                "x": round(site.frac_coords[0], 6),
                "y": round(site.frac_coords[1], 6),
                "z": round(site.frac_coords[2], 6),
                "occupancy": round(occupancy, 6),
                "label": f"{specie}{index}",
            }
        )
    space_group, _ = structure.get_space_group_info(symprec=1e-3)
    return {
        "name": name,
        "space_group": space_group,
        "lattice": {
            "a": round(lattice.a, 6),
            "b": round(lattice.b, 6),
            "c": round(lattice.c, 6),
            "alpha": round(lattice.alpha, 6),
            "beta": round(lattice.beta, 6),
            "gamma": round(lattice.gamma, 6),
        },
        "atom_sites": atom_sites,
    }


def structure_summary(structure: Structure, name: str | None = None) -> dict:
    lattice_matrix = structure.lattice.matrix.tolist()
    space_group, _ = structure.get_space_group_info(symprec=1e-3)
    return {
        "formula": structure.composition.reduced_formula,
        "density": structure.density,
        "volume": structure.volume,
        "lattice_vectors": lattice_matrix,
        "space_group": space_group,
        "warnings": [],
        "name": name,
    }


def compute_xrd(structure: Structure, two_theta_range: tuple[float, float] = (10, 90)) -> dict:
    pattern = xrd_calculator.get_pattern(structure, two_theta_range=two_theta_range)
    peaks = []
    for two_theta, intensity, d, hkls in zip(pattern.x, pattern.y, pattern.d_hkls, pattern.hkls):
        if intensity < 1e-3:
            continue
        hkl = hkls[0]["hkl"] if hkls else (0, 0, 0)
        peaks.append(
            {
                "two_theta": float(round(two_theta, 6)),
                "intensity": float(round(intensity, 6)),
                "d_spacing": float(round(d, 6)),
                "hkl": [int(i) for i in hkl],
            }
        )
    return {"peaks": peaks}


def compute_tem(structure: Structure, zone_axis: Iterable[int], *, top_n: int = 40) -> dict:
    tem = TEMCalculator(beam_direction=tuple(int(v) for v in zone_axis))
    pattern = tem.get_pattern(structure)
    sorted_pattern = pattern.sort_values("Intensity (norm)", ascending=False).head(top_n)
    reflections = []
    for _, row in sorted_pattern.iterrows():
        hkl = row["(hkl)"]
        position = row["Position"]
        d_spacing = float(row["Interplanar Spacing"])
        reflections.append(
            {
                "hkl": [int(value) for value in hkl],
                "intensity": float(row["Intensity (norm)"]),
                "g": 1.0 / d_spacing if d_spacing else 0.0,
                "position": [float(position[0]), float(position[1])],
            }
        )
    return {"reflections": reflections}


def build_fixture(phase_name: str, cif_name: str, *, zone_axes: list[tuple[int, int, int]]):
    structure = Structure.from_file(DATA_DIR / cif_name)
    model = structure_to_model(structure, phase_name)
    summary = structure_summary(structure, phase_name)
    xrd = compute_xrd(structure)
    tem_by_axis = {" ".join(map(str, axis)): compute_tem(structure, axis) for axis in zone_axes}
    default_axis_key = " ".join(map(str, zone_axes[0]))

    unique_elements = sorted({site.specie.symbol for site in structure.sites})
    base_colors = {
        "Fe": "#f87171",
        "Zr": "#38bdf8",
        "O": "#a3e635",
    }
    element_colors = [
        {"element": element, "color": base_colors.get(element, "#94a3b8")}
        for element in unique_elements
    ]

    fixture = {
        "phase": phase_name,
        "structure": model,
        "summary": summary,
        "xrd": xrd,
        "tem": tem_by_axis,
        "default_tem_axis": default_axis_key,
        "ui_config": {
            "background_color": "#0f172a",
            "atom_scale": 0.85,
            "bond_thickness": 0.12,
            "show_bonds": True,
            "default_supercell": [1, 1, 1],
            "element_colors": element_colors,
        },
    }

    out_path = FIXTURE_DIR / f"{phase_name.lower().replace(' ', '-')}.json"
    out_path.write_text(json.dumps(fixture, indent=2))
    print(f"Wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    build_fixture("Fe alpha", "Fe.cif", zone_axes=[(0, 0, 1), (1, 1, 0)])
    build_fixture("Zr alpha", "Zr-Alpha.cif", zone_axes=[(0, 0, 1), (1, 0, 0)])
