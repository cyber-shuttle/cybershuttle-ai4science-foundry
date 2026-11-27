import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
from metrics import average_errors, component_errors, convert_stress_kbar_to_gpa, magnetic_moments_or_none


def test_vasp_stress_sign_and_units():
    value = convert_stress_kbar_to_gpa([[10, 0, 0], [0, -20, 0], [0, 0, 30]])
    assert value.diagonal().tolist() == pytest.approx([-1.0, 2.0, -3.0])


def test_missing_magmom_not_zero():
    assert magnetic_moments_or_none(None, 2) is None
    assert magnetic_moments_or_none([1.2, None], 2) is None
    assert magnetic_moments_or_none([-1.2, 0.8], 2).tolist() == pytest.approx([1.2, 0.8])


def test_mae_weighted_by_components():
    pred = {
        "energy_per_atom": -2.0,
        "forces_ev_per_angstrom": [[0.10, 0, 0], [0, 0.05, 0]],
        "stress_gpa": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "magmom_mu_b": [0.4, 1.3],
    }
    ref = {
        "energy_per_atom": -2.020,
        "force": [[0, 0, 0], [0, 0, 0]],
        "stress": [[-10, 0, 0], [0, -10, 0], [0, 0, -10]],
        "magmom": [-0.5, 1.0],
    }
    rows = component_errors(pred, ref, 2)
    avg = average_errors([rows])
    assert avg["energy_mev_per_atom"]["mae"] == pytest.approx(20)
    assert avg["force_mev_per_angstrom"]["mae"] == pytest.approx(25)
    assert avg["force_mev_per_angstrom"]["count"] == 6
    assert avg["stress_gpa"]["mae"] == pytest.approx(0)
    assert avg["magmom_mu_b"]["mae"] == pytest.approx(0.2)
