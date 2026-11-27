"""
Metric helpers for CHGNet inference.

Energy: eV/atom, forces: eV/Angstrom, stress: GPa, magmom: mu_B.
MPtrj stress is raw VASP kBar, so I convert it to CHGNet's GPa convention.
MPtrj magmom is signed, while CHGNet uses absolute moments.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def convert_stress_kbar_to_gpa(stress_kbar: Any) -> np.ndarray:
    """MPtrj raw VASP stress (kBar) -> CHGNet convention (GPa)."""
    value = np.asarray(stress_kbar, dtype=float)
    if value.shape != (3, 3):
        raise ValueError(f"Expected 3x3 DFT stress, received {value.shape}")
    return -0.1 * value


def magnetic_moments_or_none(moments: Any, n_atoms: int) -> np.ndarray | None:
    """Discard a structure's magmom label if any atom has missing data."""
    if moments is None:
        return None
    try:
        if len(moments) != n_atoms or any(item is None for item in moments):
            return None
        result = np.abs(np.asarray(moments, dtype=float).reshape(-1))
    except (TypeError, ValueError):
        return None
    if len(result) != n_atoms or not np.isfinite(result).all():
        return None
    return result


def component_errors(prediction: dict, reference: dict, n_atoms: int) -> dict[str, list[float]]:
    """Return component-wise absolute errors in the same units used in the report."""
    energy_p = float(prediction["energy_per_atom"])
    energy_r = float(reference["energy_per_atom"])
    errors: dict[str, list[float]] = {
        "energy_mev_per_atom": [abs(energy_p - energy_r) * 1000],
    }

    force_p = np.asarray(prediction["forces_ev_per_angstrom"], dtype=float)
    force_r = np.asarray(reference["force"], dtype=float)
    if force_p.shape != (n_atoms, 3) or force_r.shape != (n_atoms, 3):
        raise ValueError(f"Unexpected force shape: {force_p.shape}, {force_r.shape}")
    errors["force_mev_per_angstrom"] = (1000 * np.abs(force_p - force_r)).ravel().tolist()

    stress_p = np.asarray(prediction["stress_gpa"], dtype=float)
    stress_r = convert_stress_kbar_to_gpa(reference["stress"])
    if stress_p.shape != (3, 3):
        raise ValueError(f"Unexpected predicted stress shape: {stress_p.shape}")
    errors["stress_gpa"] = np.abs(stress_p - stress_r).ravel().tolist()

    moment_r = magnetic_moments_or_none(reference.get("magmom"), n_atoms)
    if moment_r is not None:
        moment_p = np.asarray(prediction["magmom_mu_b"], dtype=float).reshape(-1)
        if moment_p.shape != moment_r.shape:
            raise ValueError("Predicted magnetic moment length differs from reference")
        errors["magmom_mu_b"] = np.abs(moment_p - moment_r).tolist()
    return errors


def average_errors(all_errors: list[dict[str, list[float]]]) -> dict[str, dict[str, float | int | None]]:
    """Compute aggregate MAE and counts."""
    keys = ("energy_mev_per_atom", "force_mev_per_angstrom", "stress_gpa", "magmom_mu_b")
    result = {}
    for key in keys:
        vals = [float(x) for row in all_errors for x in row.get(key, [])]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f"Non-finite error values in {key}")
        result[key] = {"mae": (sum(vals) / len(vals)) if vals else None, "count": len(vals)}
    return result
