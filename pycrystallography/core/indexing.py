"""Utilities for converting Miller indices and crystallographic vectors."""
from __future__ import annotations

from fractions import Fraction
from math import gcd, isclose, lcm
from typing import Iterable, Sequence, Tuple, cast

import numpy as np


MILLER_TOLERANCE = 1e-8


def _normalise_integer_vector(values: Iterable[int]) -> Tuple[int, ...]:
    """Return the tuple reduced by the greatest common divisor."""

    ints = tuple(int(v) for v in values)
    non_zero = [abs(v) for v in ints if v]
    if not non_zero:
        return ints
    divisor = non_zero[0]
    for value in non_zero[1:]:
        divisor = gcd(divisor, value)
    if divisor <= 0:
        return ints
    return tuple(v // divisor for v in ints)


def validate_miller_bravais(indices: Sequence[float]) -> None:
    """Ensure a four-index set obeys the Miller–Bravais constraint."""

    if len(indices) != 4:
        raise ValueError("Miller–Bravais notation requires four indices")
    u, v, t, _ = indices
    if not isclose(u + v + t, 0.0, abs_tol=MILLER_TOLERANCE):
        raise ValueError(
            "Invalid Miller–Bravais indices: the first three values must sum to zero",
        )


def direction_indices_to_fractional(indices: Sequence[float]) -> np.ndarray:
    """Convert a direction specification to fractional coordinates."""

    vector = np.asarray(indices, dtype=float)
    if vector.size == 3:
        return vector
    if vector.size == 4:
        validate_miller_bravais(vector.tolist())
        u, v, t, w = vector
        # Express using the three primitive lattice directions (a, b, c)
        return np.array([u - t, v - t, w], dtype=float)
    raise ValueError("Direction indices must have three or four components")


def plane_indices_to_fractional(indices: Sequence[float]) -> np.ndarray:
    """Convert plane indices to fractional reciprocal coordinates."""

    vector = np.asarray(indices, dtype=float)
    if vector.size == 3:
        return vector
    if vector.size == 4:
        validate_miller_bravais(vector.tolist())
        h, k, _i, ell = vector
        return np.array([h, k, ell], dtype=float)
    raise ValueError("Plane indices must have three or four components")


def direction_to_cartesian(structure, indices: Sequence[float]) -> np.ndarray:
    """Convert a direction into a Cartesian vector using a structure's lattice."""

    fractional = direction_indices_to_fractional(indices)
    cart = np.asarray(structure.lattice.get_cartesian_coords(fractional), dtype=float)
    if not np.any(cart):
        raise ValueError("Direction indices produced a zero-length vector")
    return cart


def plane_normal_to_cartesian(structure, indices: Sequence[float]) -> np.ndarray:
    """Return the Cartesian plane normal corresponding to given indices."""

    fractional = plane_indices_to_fractional(indices)
    reciprocal = structure.lattice.reciprocal_lattice
    cart = np.asarray(reciprocal.get_cartesian_coords(fractional), dtype=float)
    if not np.any(cart):
        raise ValueError("Plane indices produced a zero-length normal vector")
    return cart


def indices_to_cartesian(structure, *, kind: str, indices: Sequence[float]) -> np.ndarray:
    """Dispatch to the appropriate Cartesian conversion helper."""

    if kind == "direction":
        return direction_to_cartesian(structure, indices)
    if kind == "plane":
        return plane_normal_to_cartesian(structure, indices)
    raise ValueError(f"Unknown index kind '{kind}'")


def fractional_to_integer_indices(
    fractional: Sequence[float],
    *,
    max_denominator: int = 24,
    round_tol: float = 1e-6,
) -> Tuple[int, int, int]:
    """Convert fractional coordinates to the smallest integer representation."""

    fractions: list[Fraction] = []
    for value in fractional:
        if abs(value) < round_tol:
            fractions.append(Fraction(0, 1))
        else:
            fractions.append(Fraction(float(value)).limit_denominator(max_denominator))
    denominators = [frac.denominator for frac in fractions]
    overall = 1
    for denominator in denominators:
        overall = lcm(overall, denominator)
    integers = [frac.numerator * (overall // frac.denominator) for frac in fractions]
    normalised = _normalise_integer_vector(integers)
    return cast(Tuple[int, int, int], tuple(int(v) for v in normalised))


def direction_three_to_four(indices: Sequence[int]) -> Tuple[int, int, int, int]:
    """Convert three-index direction integers to four-index notation."""

    u3, v3, w3 = indices
    u = 2 * u3 - v3
    v = 2 * v3 - u3
    t = -(u + v)
    w = 3 * w3
    return cast(Tuple[int, int, int, int], _normalise_integer_vector((u, v, t, w)))


def plane_three_to_four(indices: Sequence[int]) -> Tuple[int, int, int, int]:
    """Convert three-index plane indices to Miller–Bravais notation."""

    h3, k3, l3 = indices
    i = -(h3 + k3)
    return cast(Tuple[int, int, int, int], _normalise_integer_vector((h3, k3, i, l3)))


def ensure_column_vector(vector: np.ndarray) -> np.ndarray:
    """Return a flattened 3-vector from arbitrary shaped input."""

    arr = np.asarray(vector, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError("Expected a three-component vector")
    return arr


__all__ = [
    "MILLER_TOLERANCE",
    "direction_indices_to_fractional",
    "direction_three_to_four",
    "direction_to_cartesian",
    "ensure_column_vector",
    "fractional_to_integer_indices",
    "indices_to_cartesian",
    "plane_indices_to_fractional",
    "plane_normal_to_cartesian",
    "plane_three_to_four",
    "validate_miller_bravais",
]
