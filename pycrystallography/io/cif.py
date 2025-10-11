"""CIF ingestion helpers backed by :mod:`pymatgen`."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


@dataclass(slots=True)
class StructureSummary:
    """Container holding a structure plus normalised metadata."""

    path: Path
    structure: Structure
    space_group: int
    lattice_parameters: Dict[str, float]

    def validate_metadata(self, metadata: Mapping[str, float | int | None]) -> List[str]:
        """Validate user-supplied metadata against the CIF contents.

        Parameters
        ----------
        metadata
            Mapping containing optional keys ``space_group`` and lattice
            parameters ``a``, ``b``, ``c``, ``alpha``, ``beta`` and ``gamma``.

        Returns
        -------
        list[str]
            A list of warning messages for any mismatches. The CIF is treated as
            the source of truth; the method never mutates the underlying
            structure.
        """

        warnings: List[str] = []
        if "space_group" in metadata and metadata["space_group"] is not None:
            expected = int(metadata["space_group"])
            if expected != self.space_group:
                warnings.append(
                    "Space group mismatch: CIF reports"
                    f" {self.space_group} but metadata supplied {expected}."
                )
        keys = ("a", "b", "c", "alpha", "beta", "gamma")
        for key in keys:
            if key not in metadata or metadata[key] is None:
                continue
            cif_value = self.lattice_parameters[key]
            supplied = float(metadata[key])
            if abs(cif_value - supplied) > 1e-3:
                warnings.append(
                    f"Lattice parameter '{key}' mismatch: CIF {cif_value:.4f}"
                    f" vs supplied {supplied:.4f}."
                )
        return warnings


def _normalise_lattice(structure: Structure) -> Dict[str, float]:
    lattice = structure.lattice
    return {
        "a": float(lattice.a),
        "b": float(lattice.b),
        "c": float(lattice.c),
        "alpha": float(lattice.alpha),
        "beta": float(lattice.beta),
        "gamma": float(lattice.gamma),
    }


def load_cif_structure(
    path: str | Path, *, symprec: Optional[float] = 1e-5
) -> StructureSummary:
    """Load a CIF file and return a :class:`StructureSummary`.

    The loader always resolves the provided path, ensuring deterministic caching
    downstream. It also normalises the space group using
    :class:`~pymatgen.symmetry.analyzer.SpacegroupAnalyzer`.
    """

    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"CIF file does not exist: {file_path}")
    structure = Structure.from_file(str(file_path))
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    space_group_number = int(analyzer.get_space_group_number())
    summary = StructureSummary(
        path=file_path,
        structure=structure,
        space_group=space_group_number,
        lattice_parameters=_normalise_lattice(structure),
    )
    return summary


class StructureCache:
    """Lightweight in-memory cache keyed by resolved CIF path."""

    def __init__(self) -> None:
        self._cache: Dict[Path, StructureSummary] = {}

    def get(self, path: str | Path, *, symprec: Optional[float] = 1e-5) -> StructureSummary:
        file_path = Path(path).expanduser().resolve()
        if file_path not in self._cache:
            self._cache[file_path] = load_cif_structure(file_path, symprec=symprec)
        return self._cache[file_path]

    def preload(self, paths: Iterable[str | Path], *, symprec: Optional[float] = None) -> None:
        for path in paths:
            self.get(path, symprec=symprec)


__all__ = ["StructureCache", "StructureSummary", "load_cif_structure"]
